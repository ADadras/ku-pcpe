import argparse
from .evaluate import evaluate


def main():
    p = argparse.ArgumentParser(
        description="KU-PCPE saved-prediction evaluation, CPU only"
    )
    p.add_argument("--annotations", required=True)
    p.add_argument("--replay", required=True)
    p.add_argument("--expected", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    evaluate(a.annotations, a.replay, a.expected, a.output, a.workers, a.resume)


if __name__ == "__main__":
    main()
