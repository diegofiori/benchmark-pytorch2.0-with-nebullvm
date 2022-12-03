import argparse
import json

import numpy as np
import torch
import torchvision.models as models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--nebullvm", "-n", action="store_true")
    parser.add_argument("--max_autotune", action="store_true")
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()
    model = models.resnet50(pretrained=True)
    input_data = [((torch.randn(args.batch_size, 3, 224, 224), ), torch.zeros(args.batch_size))]
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        input_data = [((x[0].cuda(), ), y.cuda()) for x, y in input_data]
        if args.half:
            model = model.half()
            input_data = [((x[0].half(), ), y.half()) for x, y in input_data]

    if args.nebullvm:
        from nebullvm import optimize_model
        optimized_model = optimize_model(model, input_data, store_latencies=True, metric_drop_ths=100., optimization_time="unconstrained")
    else:
        mode = "max-autotune" if args.max_autotune else "default"
        optimized_model = torch.compile(model, mode=mode)

    new_input_data = [torch.randn(args.batch_size, 3, 224, 224) for _ in range(100)]
    if torch.cuda.is_available():
        new_input_data = [x.cuda() for x in new_input_data]
        if args.half:
            new_input_data = [x.half() for x in new_input_data]

    times = []
    with torch.no_grad():
        for i, x in enumerate(new_input_data):
            if i < 10:
                optimized_model(x)  # warmup
                continue
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            optimized_model(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    print(f"Average inference time: {sum(times) / len(times)} ms")
    results = {
        "latency": sum(times) / len(times),
        "median": float(np.median(times)),
        "std": float(np.std(times)),
        "min": min(times),
        "max": max(times),
    }
    if not args.nebullvm:
        results["mode"] = mode
        filename = f"results_torch_{args.batch_size}_{mode}.json"
        if args.half:
            filename = f"results_torch_{args.batch_size}_{mode}_half.json"
    else:
        filename = f"results_nebullvm_{args.batch_size}.json"

    with open(filename, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
