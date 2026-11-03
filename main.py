from time_study.config import build_arg_parser, load_config


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args)

    from time_study.video_app import run_video_app

    return run_video_app(config)

if __name__ == "__main__":
    raise SystemExit(main())
