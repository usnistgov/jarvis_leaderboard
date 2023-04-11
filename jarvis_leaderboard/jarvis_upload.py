import argparse
import os
import sys
import requests

parser = argparse.ArgumentParser(description="Add data to JARVIS-Leaderboard.")


parser.add_argument(
    "--upstream_repo_name",
    default="jarvis_leaderboard",
    help="Upstream/source repo name",
)

parser.add_argument(
    "--upstream_repo_username",
    default="usnistgov",
    help="Upstream/source repo user name",
)

parser.add_argument(
    "--github_username",
    default="knc6",
    help="Your GitHub username",
)


parser.add_argument(
    "--your_benchmark_directory",
    default="my_example_benchmark",
    help="Your benchmark to be added.",
)


def upload():
    args = parser.parse_args(sys.argv[1:])
    upstream_repo_name = args.upstream_repo_name
    upstream_repo_username = args.upstream_repo_username
    username = args.github_username
    your_benchmark_directory = args.your_benchmark_directory
    cwd = os.getcwd()
    print("username", username)
    forked_url = "https://github.com/" + username + "/" + upstream_repo_name
    print("forked_url", forked_url)
    response = requests.get(forked_url)
    if response.status_code > 400:
        cmd = (
            "curl -u "
            + username
            + " https://api.github.com/repos/"
            + upstream_repo_username
            + "/"
            + upstream_repo_name
            + "/forks -d ''"
        )
        print(cmd)
        os.system(cmd)
    if not os.path.exists(upstream_repo_name):
        cmd = "git clone " + forked_url + ".git"
        print(cmd)
        os.system(cmd)
    if os.path.exists(your_benchmark_directory):
        print("Note: adding to existing directory.")
    cmd = (
        "rsync -r "
        + your_benchmark_directory
        + " jarvis_leaderboard/jarvis_leaderboard/benchmarks"
    )
    print(cmd)
    os.system(cmd)
    # cmd='cd '+upstream_repo_name
    os.chdir(upstream_repo_name)
    add_dir = "jarvis_leaderboard/benchmarks/" + your_benchmark_directory
    cmd = "ls ./" + add_dir
    print(cmd)
    os.system(cmd)

    cmd = "git add ./" + add_dir + "/*"
    print(cmd)
    os.system(cmd)
    cmd = "git commit"
    print(cmd)
    os.system(cmd)
    cmd = "git push"
    print(cmd)
    os.system(cmd)

    cmd = "python jarvis_leaderboard/rebuild.py"
    print(cmd)
    os.system(cmd)

    cmd = (
        "curl -u "
        + username
        + " -d "
        + "'"
        + '{"title":"Adding new benchmark","base":"develop", "head":"'
        + username
        + ':main"}'
        + "'"
        + " https://api.github.com/repos/usnistgov/jarvis_leaderboard/pulls"
    )
    print(cmd)
    os.system(cmd)

    os.chdir(cwd)


if __name__ == "__main__":
    upload()
# export upstream_repo_name=jarvis_leaderboard
# export upstream_repo_username=usnistgov
# export my_user_name=knc6
# curl -u $my_user_name https://api.github.com/repos/$upstream_repo_username/$upstream_repo_name/forks -d ''
