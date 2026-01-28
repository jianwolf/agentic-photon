#!/usr/bin/env python3
"""Photon: Intelligent news analysis pipeline powered by PydanticAI agents.

This CLI tool monitors RSS feeds, classifies news importance using AI,
and generates detailed analysis reports for important stories.

Commands:
    run         Execute the analysis pipeline (once or continuously)
    status      Show configuration and database statistics
    recent      Display recent important stories
    analyze     Manually analyze a specific story
    digest      Generate a digest from existing markdown reports

Examples:
    python main.py run                    # Single run
    python main.py run -c                 # Continuous polling
    python main.py run --lang en          # English output
    python main.py status                 # Show config
    python main.py recent --hours 48      # Last 48 hours
    python main.py analyze --title "..." --force

Environment:
    GEMINI_API_KEY: Required for AI agents
    See config.py for all configuration options
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from config import Config
from database import Database
from observability.logging import setup_logging


def cmd_run(args: argparse.Namespace, config: Config) -> int:
    """Execute the news analysis pipeline.

    Args:
        args: Parsed command line arguments
        config: Application configuration

    Returns:
        Exit code (0 for success)
    """
    from pipeline import run_once, run_continuous
    from mlx_server import MLXServerManager

    # Override config with CLI arguments
    if args.lang:
        config.language = args.lang
    if args.interval:
        config.poll_interval_seconds = args.interval

    max_stories = getattr(args, 'max_stories', 0) or 0
    classifier_model = getattr(args, 'classifier_model', None)
    mlx_port = getattr(args, 'mlx_port', 8080)

    logger = logging.getLogger(__name__)

    # Start MLX server for local classifier
    mlx_server = None
    if classifier_model:
        logger.info("Starting MLX server for classifier | model=%s port=%d", classifier_model, mlx_port)
        mlx_server = MLXServerManager(model=classifier_model, port=mlx_port)
        mlx_server.start()  # Blocks until server is ready
        # Update config to use local classifier (format: openai:{model_name}@{base_url})
        config.classifier_model = f"openai:{classifier_model}@http://127.0.0.1:{mlx_port}/v1"

    try:
        if args.continuous:
            logger.info("Starting continuous mode...")
            try:
                asyncio.run(run_continuous(config, max_stories=max_stories))
            except KeyboardInterrupt:
                logger.info("Stopped by user (Ctrl+C)")
            return 0
        else:
            stats = asyncio.run(run_once(config, max_stories=max_stories))
            logger.info("Run complete | stats=%s", json.dumps(stats))
            return 0
    except KeyboardInterrupt:
        logger.info("Stopped by user (Ctrl+C)")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error("Pipeline failed | error=%s type=%s", e, type(e).__name__, exc_info=True)
        return 1
    finally:
        if mlx_server:
            mlx_server.stop()


def cmd_status(args: argparse.Namespace, config: Config) -> int:
    """Display configuration and database statistics.

    Args:
        args: Parsed command line arguments
        config: Application configuration

    Returns:
        Exit code (0 for success)
    """
    with Database(config.db_path) as db:
        db_stats = db.stats()

    status = {
        "config": {
            "language": config.language,
            "classifier_model": config.classifier_model,
            "researcher_model": config.researcher_model,
            "researcher_model_pro": config.researcher_model_pro,
            "feeds": len(config.rss_urls),
            "max_age_hours": config.max_age_hours,
            "poll_interval": config.poll_interval_seconds,
            "embedded_stories": db_stats["embedded"],
            "enable_logfire": config.enable_logfire,
        },
        "database": {
            "path": str(config.db_path),
            "total_stories": db_stats["total"],
            "important_stories": db_stats["important"],
        },
    }

    print(json.dumps(status, indent=2))
    return 0


def cmd_recent(args: argparse.Namespace, config: Config) -> int:
    """Display recent important stories.

    Args:
        args: Parsed command line arguments
        config: Application configuration

    Returns:
        Exit code (0 for success)
    """
    with Database(config.db_path) as db:
        stories = db.recent(hours=args.hours)

    if not stories:
        print(f"No important stories in the last {args.hours} hours.")
        return 0

    print(f"\n=== Important Stories (last {args.hours} hours) ===\n")

    for story in stories:
        pub_date = datetime.fromtimestamp(story["pub_date"])
        processed = datetime.fromtimestamp(story["processed_at"])

        print(f"📰 {story['title']}")
        print(f"   Published: {pub_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Processed: {processed.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Source: {story['source_url']}")

        if story.get("summary"):
            summary = story["summary"]
            if len(summary) > 200:
                summary = summary[:200] + "..."
            print(f"   Summary: {summary}")

        print()

    return 0


def cmd_analyze(args: argparse.Namespace, config: Config) -> int:
    """Analyze a specific URL or title.

    Args:
        args: Parsed command line arguments
        config: Application configuration

    Returns:
        Exit code (0 for success)
    """
    from datetime import timezone
    from models.story import Story
    from agents.classifier import ClassifierAgent
    from agents.researcher import ResearcherAgent, StoryContext
    from tools.fetch import fetch_article

    async def analyze_story():
        # Create a story from the input
        story = Story(
            title=args.title or args.url,
            description=args.description or "",
            pub_date=datetime.now(timezone.utc),
            source_url=args.url or "manual",
            article_url=args.url or "",
        )

        # Override language if specified
        if args.lang:
            config.language = args.lang

        # Classify
        classifier = ClassifierAgent(config)
        classification = await classifier.classify(story)

        print("\n=== Classification ===")
        print(f"Important: {classification.is_important}")
        print(f"Category: {classification.category.value}")
        print(f"Confidence: {classification.confidence:.2f}")
        if classification.reasoning:
            print(f"Reasoning: {classification.reasoning}")

        if classification.is_important or args.force:
            # Fetch article content if URL provided
            article_content = ""
            if args.url:
                result = await fetch_article(args.url)
                if result.success:
                    article_content = result.content

            # Build context and analyze
            story_context = StoryContext(
                story=story,
                classification=classification,
                article_content=article_content,
                related_stories="",
            )

            researcher = ResearcherAgent(config)
            report = await researcher.analyze(story_context)

            print("\n=== Analysis ===")
            print(f"\n{report.summary}")

            if report.thought:
                print("\n--- Notes ---")
                print(report.thought)

            if report.key_points:
                print("\n--- Key Points ---")
                for point in report.key_points:
                    print(f"• {point}")

    asyncio.run(analyze_story())
    return 0


def _latest_digest_mtime(reports_dir: Path) -> float | None:
    """Return the latest digest modification timestamp in a directory."""
    digest_paths = [path for path in reports_dir.glob("*_digest.md") if path.is_file()]
    if not digest_paths:
        return None
    return max(path.stat().st_mtime for path in digest_paths)


def _select_report_paths(
    reports_dir: Path,
    exclude_compare: bool,
    since_hours: float | None,
    since_last_digest: bool,
    max_reports: int,
) -> list[Path]:
    """Select report markdown files for digest generation."""
    if not reports_dir.exists():
        return []

    report_paths = [path for path in reports_dir.glob("*.md") if path.is_file()]
    report_paths = [path for path in report_paths if not path.name.endswith("_digest.md")]

    if exclude_compare:
        report_paths = [
            path
            for path in report_paths
            if not (path.name.endswith("_flash.md") or path.name.endswith("_pro.md"))
        ]

    cutoff = None
    if since_hours is not None:
        cutoff = time.time() - (since_hours * 3600)
    elif since_last_digest:
        cutoff = _latest_digest_mtime(reports_dir)

    if cutoff is not None:
        report_paths = [path for path in report_paths if path.stat().st_mtime > cutoff]

    report_paths.sort(key=lambda path: (path.stat().st_mtime, path.name))

    if max_reports > 0 and len(report_paths) > max_reports:
        report_paths = report_paths[-max_reports:]

    return report_paths


def cmd_digest(args: argparse.Namespace, config: Config) -> int:
    """Generate a digest from existing markdown reports."""
    from agents.summarizer import SummarizerAgent, render_digest_markdown
    from notifications import save_digest_report

    if args.lang:
        config.language = args.lang
    if args.summary_model:
        config.summary_model = args.summary_model

    reports_dir = Path(args.reports_dir) if args.reports_dir else config.reports_dir
    since_last_digest = not args.all and args.since_hours is None
    report_paths = _select_report_paths(
        reports_dir=reports_dir,
        exclude_compare=args.exclude_compare,
        since_hours=args.since_hours,
        since_last_digest=since_last_digest,
        max_reports=args.max_reports,
    )

    if not report_paths:
        print("No report markdown files selected for digest.")
        return 0

    if args.dry_run:
        print(f"Selected {len(report_paths)} report(s) from {reports_dir}:")
        for path in report_paths:
            print(f"- {path.name}")
        return 0

    if error := config.validate():
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1

    async def run_digest() -> tuple[Path | None, int, int]:
        summarizer = SummarizerAgent(config)
        digest, input_tokens, output_tokens = await summarizer.summarize_paths(report_paths)
        digest_md = render_digest_markdown(digest, report_paths)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(digest_md, encoding="utf-8")
            return output_path, input_tokens, output_tokens
        digest_path = await save_digest_report(digest_md, reports_dir)
        return digest_path, input_tokens, output_tokens

    digest_path, input_tokens, output_tokens = asyncio.run(run_digest())
    if digest_path:
        print(
            f"Digest saved: {digest_path} "
            f"(reports={len(report_paths)} tokens={input_tokens}/{output_tokens})"
        )
    else:
        print("Digest generated but failed to save.", file=sys.stderr)
        return 1
    return 0


def cmd_compare(args: argparse.Namespace, config: Config) -> int:
    """Compare researcher models on a sample of stories.

    Args:
        args: Parsed command line arguments
        config: Application configuration

    Returns:
        Exit code (0 for success)
    """
    from pathlib import Path
    from pipeline import compare_models
    from mlx_server import MLXServerManager

    # Override config with CLI arguments
    if args.lang:
        config.language = args.lang
    if args.flash_model:
        config.researcher_model = args.flash_model
    if args.pro_model:
        config.researcher_model_pro = args.pro_model

    output_dir = Path(args.output_dir) if args.output_dir else None
    limit = max(1, args.limit)
    classifier_model = getattr(args, "classifier_model", None)
    mlx_port = getattr(args, "mlx_port", 8080)

    logger = logging.getLogger(__name__)

    # Optional MLX server for local classifier
    mlx_server = None
    if classifier_model:
        logger.info("Starting MLX server for classifier | model=%s port=%d", classifier_model, mlx_port)
        mlx_server = MLXServerManager(model=classifier_model, port=mlx_port)
        mlx_server.start()  # Blocks until server is ready
        config.classifier_model = f"openai:{classifier_model}@http://127.0.0.1:{mlx_port}/v1"

    try:
        result = asyncio.run(compare_models(config, limit=limit, output_dir=output_dir))
        print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        logger.info("Stopped by user (Ctrl+C)")
        return 130
    except Exception as e:
        logger.error("Comparison failed | error=%s type=%s", e, type(e).__name__, exc_info=True)
        return 1
    finally:
        if mlx_server:
            mlx_server.stop()


def main() -> int:
    """Main entry point.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Photon: Intelligent news analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run the analysis pipeline")
    run_parser.add_argument(
        "-c", "--continuous",
        action="store_true",
        help="Run continuously with polling",
    )
    run_parser.add_argument(
        "--lang",
        choices=["zh", "en"],
        help="Output language (default: zh)",
    )
    run_parser.add_argument(
        "--interval",
        type=int,
        help="Poll interval in seconds (continuous mode)",
    )
    run_parser.add_argument(
        "--max-stories",
        type=int,
        default=0,
        help="Max important stories to analyze (0 = unlimited, selects latest by date)",
    )
    run_parser.add_argument(
        "--classifier-model",
        type=str,
        default="mlx-community/Ministral-3-3B-Instruct-2512",
        help="Local MLX model for classification (default: Ministral-3B)",
    )
    run_parser.add_argument(
        "--mlx-port",
        type=int,
        default=8080,
        help="Port for local MLX server (default: 8080)",
    )

    # status command
    subparsers.add_parser("status", help="Show configuration and statistics")

    # recent command
    recent_parser = subparsers.add_parser("recent", help="Show recent important stories")
    recent_parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Look back N hours (default: 24)",
    )

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a specific story")
    analyze_parser.add_argument(
        "--url",
        help="URL of the story",
    )
    analyze_parser.add_argument(
        "--title",
        help="Title of the story",
    )
    analyze_parser.add_argument(
        "--description",
        help="Description or summary",
    )
    analyze_parser.add_argument(
        "--lang",
        choices=["zh", "en"],
        help="Output language",
    )
    analyze_parser.add_argument(
        "--force",
        action="store_true",
        help="Force analysis even if not classified as important",
    )

    # digest command
    digest_parser = subparsers.add_parser("digest", help="Generate a digest from reports")
    digest_parser.add_argument(
        "--reports-dir",
        help="Directory containing markdown reports (default: config REPORTS_DIR)",
    )
    digest_parser.add_argument(
        "--lang",
        choices=["zh", "en"],
        help="Output language",
    )
    digest_parser.add_argument(
        "--summary-model",
        help="Override summary model (default: config SUMMARY_MODEL)",
    )
    digest_parser.add_argument(
        "--since-hours",
        type=float,
        help="Only include reports modified within the last N hours",
    )
    digest_parser.add_argument(
        "--all",
        action="store_true",
        help="Include all reports (ignore since-hours and last digest)",
    )
    digest_parser.add_argument(
        "--max-reports",
        type=int,
        default=0,
        help="Max reports to include (0 = unlimited)",
    )
    digest_parser.add_argument(
        "--exclude-compare",
        action="store_true",
        help="Exclude *_flash.md and *_pro.md comparison reports",
    )
    digest_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected reports and exit without calling the model",
    )
    digest_parser.add_argument(
        "--output",
        help="Write digest markdown to this path instead of reports directory",
    )

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare flash vs pro researcher outputs")
    compare_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of stories to compare (default: 10)",
    )
    compare_parser.add_argument(
        "--lang",
        choices=["zh", "en"],
        help="Output language",
    )
    compare_parser.add_argument(
        "--output-dir",
        help="Directory to save comparison reports (default: reports/compare/<timestamp>)",
    )
    compare_parser.add_argument(
        "--flash-model",
        help="Override flash researcher model (default: config.researcher_model)",
    )
    compare_parser.add_argument(
        "--pro-model",
        help="Override pro researcher model (default: config.researcher_model_pro)",
    )
    compare_parser.add_argument(
        "--classifier-model",
        type=str,
        default="",
        help="Optional local MLX model for classification (when set, starts MLX server)",
    )
    compare_parser.add_argument(
        "--mlx-port",
        type=int,
        default=8080,
        help="Port for local MLX server (default: 8080)",
    )

    args = parser.parse_args()

    # Load configuration
    config = Config.load()

    # Setup logging
    setup_logging(config, verbose=args.verbose)

    # Validate configuration for commands that need it
    if args.command in ("run", "analyze", "compare"):
        error = config.validate()
        if error:
            print(f"Configuration error: {error}", file=sys.stderr)
            return 1

    # Route to command handler
    commands = {
        "run": cmd_run,
        "status": cmd_status,
        "recent": cmd_recent,
        "analyze": cmd_analyze,
        "digest": cmd_digest,
        "compare": cmd_compare,
    }

    if args.command in commands:
        if args.command == "analyze" and not args.url and not args.title:
            print("Error: --url or --title is required", file=sys.stderr)
            return 1
        try:
            return commands[args.command](args, config)
        except Exception as e:
            logging.getLogger(__name__).error("Command failed | cmd=%s error=%s", args.command, e, exc_info=True)
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
