import os
import json
import datetime
import shutil
import tarfile
from collections import defaultdict

from huggingface_hub import hf_hub_download

VERSION_REFERENCE_PATH = None

LATEST_RAW_REPLAY_REVISION = "v3"
LATEST_PARSED_REPLAY_REVISION = "v3"
LATEST_TEAMS_REVISION = "v3"
LATEST_USAGE_STATS_REVISION = "v1"

METAMON_CACHE_DIR = os.getenv('METAMON_CACHE_DIR', '/tmp/metamon_cache')


def _update_version_reference(key: str, name: str, version: str):
    """Maintains a version_reference.json file in the METAMON_CACHE_DIR.

    Records the version of each dataset that is currently active.
    """
    if VERSION_REFERENCE_PATH is None:
        return

    version_reference = defaultdict(dict)
    if os.path.exists(VERSION_REFERENCE_PATH):
        with open(VERSION_REFERENCE_PATH, "r") as f:
            existing_version_reference = json.load(f)
        version_reference.update(existing_version_reference)

    version_reference[key][
        name
    ] = f"version {version}, downloaded {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    with open(VERSION_REFERENCE_PATH, "w") as f:
        json.dump(dict(version_reference), f)


def get_active_dataset_versions() -> dict:
    """Get the current version of a dataset."""
    with open(VERSION_REFERENCE_PATH, "r") as f:
        version_reference = json.load(f)
    return version_reference


def download_parsed_replays(
    battle_format: str,
    version: str = LATEST_PARSED_REPLAY_REVISION,
    force_download: bool = False,
) -> str:
    """Download the parsed replays for a given battle format.

    Args:
        battle_format: Showdown battle format (e.g. "gen1ou")
        version: Version of the dataset to download. Corresponds to revisions on the
            Hugging Face Hub. Defaults to the latest version.
        force_download: If True, download the dataset even if a previous version
            already exists in the cache.

    Returns:
        The path to the dataset on disk.
    """
    if METAMON_CACHE_DIR is None:
        raise ValueError("METAMON_CACHE_DIR environment variable is not set")
    parsed_replay_dir = os.path.join(METAMON_CACHE_DIR, "parsed-replays")
    tar_path = os.path.join(parsed_replay_dir, f"{battle_format}.tar.gz")
    out_path = os.path.join(parsed_replay_dir, battle_format)
<<<<<<< HEAD
    print(out_path)
=======
>>>>>>> main
    if os.path.exists(out_path):
        if not force_download:
            return out_path
        print(f"Clearing existing dataset at {out_path}...")
        shutil.rmtree(out_path)
    hf_hub_download(
        cache_dir=os.path.join(METAMON_CACHE_DIR, "parsed-replays"),
        repo_id="jakegrigsby/metamon-parsed-replays",
        filename=f"{battle_format}.tar.gz",
        local_dir=os.path.join(METAMON_CACHE_DIR, "parsed-replays"),
        revision=version,
        repo_type="dataset",
    )
    with tarfile.open(tar_path) as tar:
        print(f"Extracting {tar_path}...")
        if version > "v1":
            extract_path = parsed_replay_dir
        else:
            extract_path = out_path
        tar.extractall(path=extract_path)
    os.remove(tar_path)
    _update_version_reference("parsed-replays", battle_format, version)
    return out_path


def download_teams(
    battle_format: str,
    set_name: str,
    version: str = LATEST_TEAMS_REVISION,
    force_download: bool = False,
) -> str:
    """Download the teams for a given battle format and set name.

    Args:
        battle_format: Showdown battle format (e.g. "gen1ou")
        set_name: Name of the team set to download.
        version: Version of the dataset to download. Corresponds to revisions on the
            Hugging Face Hub. Defaults to the latest version.
        force_download: If True, download the dataset even if a previous version
            already exists in the cache.

    Returns:
        The path to the dataset on disk.
    """
    if METAMON_CACHE_DIR is None:
        raise ValueError("METAMON_CACHE_DIR environment variable is not set")

    teams_dir = os.path.join(METAMON_CACHE_DIR, "teams", set_name)
    tar_path = os.path.join(teams_dir, f"{battle_format}.tar.gz")
    extract_path = os.path.join(teams_dir, battle_format)
    if os.path.exists(extract_path):
        if not force_download:
            return extract_path
        print(f"Clearing existing dataset at {extract_path}...")
        shutil.rmtree(extract_path)
    hf_hub_download(
        cache_dir=os.path.join(METAMON_CACHE_DIR, "teams", set_name),
        repo_id="jakegrigsby/metamon-teams",
        filename=f"{set_name}/{battle_format}.tar.gz",
        local_dir=os.path.join(METAMON_CACHE_DIR, "teams"),
        revision=version,
        repo_type="dataset",
    )
    with tarfile.open(tar_path) as tar:
        print(f"Extracting {tar_path}...")
        tar.extractall(path=os.path.dirname(extract_path))
    os.remove(tar_path)
    _update_version_reference("teams", f"{set_name}/{battle_format}", version)
    return extract_path

    import argparse
    from termcolor import colored

    parser = argparse.ArgumentParser(
        description=f"""
Metamon Dataset Downloader

This tool downloads and manages Metamon datasets from Hugging Face Hub.
Available datasets include:
    - raw-replays: Unprocessed Showdown replays (stripped of usernames/chat)
    - parsed-replays: RL-compatible version of replays with reconstructed player actions  
    - revealed-teams: Teams that were revealed during replay battles
    - replay-stats: Statistics generated from revealed teams
    - usage-stats: Team composition stats from Showdown
    - teams: Various team sets (competitive, paper_variety, paper_replays, modern_replays)

Examples:
    # Download all team files for Gen 1-4 OU
    python -m metamon.data.download teams --formats gen1ou gen2ou gen3ou gen4ou

    # Download parsed replays for Gen 1 UU  
    python -m metamon.data.download parsed-replays --formats gen1uu

    # Download (anonymized) Showdown replay logs (all formats)
    python -m metamon.data.download raw-replays

Note: Requires METAMON_CACHE_DIR environment variable to be set.

The cache directory is currently: {colored(METAMON_CACHE_DIR or 'NOT SET', 'red')}
For current dataset versions, see `get_active_dataset_versions()` or run:
    python -m metamon.download check-versions
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=[
            "raw-replays",
            "parsed-replays",
            "revealed-teams",
            "replay-stats",
            "teams",
            "usage-stats",
            "check-versions",
        ],
        help="""
`check-versions`: Display the current versions of the datasets.

Dataset to download:
    raw-replays: Unprocessed Showdown replays (stripped of usernames/chat)
    parsed-replays: RL-compatible version of replays with reconstructed player actions
    revealed-teams: Teams that were revealed during battles
    replay-stats: Statistics generated from revealed teams. Used to predict team sets.
    teams: Various team sets (competitive, paper_variety, paper_replays)
""",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        type=str,
        default=SUPPORTED_BATTLE_FORMATS,
        help="""
Battle formats to download. Defaults to all Gen 1-4 formats (OU, UU, NU, Ubers).
Examples:
    --formats gen1ou gen2ou    # Only Gen 1-2 OU
    --formats gen3uu gen4uu    # Only Gen 3-4 UU
""",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="""
Specific version to download. Defaults to latest version.
Available versions:
    raw-replays: v1, v2, v3
    parsed-replays: v0 (deprecated) v1, v2, v3-beta
    teams: v0, v1, v2, v3
    usage-stats: v0, v1
    
    The huggingface READMEs have changelogs.
""",
    )
    args = parser.parse_args()

    if args.dataset == "raw-replays":
        version = args.version or LATEST_RAW_REPLAY_REVISION
        download_raw_replays(version=version)
    elif args.dataset == "parsed-replays":
        version = args.version or LATEST_PARSED_REPLAY_REVISION
        if args.formats is None:
            raise ValueError("Must specify at least one battle format (e.g., gen1ou)")
        for format in args.formats:
            download_parsed_replays(format, version=version, force_download=True)
    elif args.dataset == "revealed-teams":
        version = args.version or LATEST_PARSED_REPLAY_REVISION
        download_revealed_teams(version=version, force_download=True)
    elif args.dataset == "replay-stats":
        version = args.version or LATEST_PARSED_REPLAY_REVISION
        download_replay_stats(version=version, force_download=True)
    elif args.dataset == "usage-stats":
        version = args.version or LATEST_USAGE_STATS_REVISION
        if args.formats is None:
            raise ValueError("Must specify at least one battle format (e.g., gen1ou)")
        generations = set(
            metamon.backend.format_to_gen(format) for format in args.formats
        )
        for gen in generations:
            download_usage_stats(gen=gen, version=version, force_download=True)
    elif args.dataset == "teams":
        version = args.version or LATEST_TEAMS_REVISION
        if args.formats is None:
            raise ValueError("Must specify at least one set name (e.g., gen1ou)")
        set_names = ["competitive", "paper_variety", "paper_replays"]
        if version > "v0":
            set_names.append("modern_replays")
        for set_name in set_names:
            for format in args.formats:
                if "ou" not in format and "replays" in set_name:
                    # only OU tiers have replay sets currently
                    continue
                if "gen9" in format and "paper" in set_name:
                    # gen 9 was not supported
                    continue
                download_teams(
                    battle_format=format,
                    set_name=set_name,
                    version=version,
                    force_download=True,
                )
    elif args.dataset == "check-versions":
        print(colored("\nActive dataset versions:", "red"))
        print(f"Cache Dir: {METAMON_CACHE_DIR}\n")
        print_version_tree(get_active_dataset_versions())
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")