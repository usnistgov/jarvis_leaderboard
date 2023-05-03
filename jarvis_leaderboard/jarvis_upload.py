#!/home/knc6/.conda/envs/mer/bin/python
import argparse
import os
import sys
import requests
import time
from jarvis_leaderboard.rebuild import rebuild_pages

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
    default="not_available",
    help="Your GitHub username",
)


parser.add_argument(
    "--your_contribution_directory",
    default="my_example_contribution",
    help="Your contribution to be added.",
)


def upload():
    print("Make sure GitHub credentials are added\nChecking:\n")
    cmd = "git config --list > ghout"
    os.system(cmd)

    args = parser.parse_args(sys.argv[1:])
    username = args.github_username
    f = open("ghout", "r")
    lines = f.read().splitlines()
    f.close()
    username = ""
    passwd = ""
    for i in lines:
        if "user.name" in i:
            username = i.split("=")[1]
        if "user.password" in i:
            passwd = i.split("=")[1]
            print("passwd/token", passwd)
    cmd = "rm ghout"
    os.system(cmd)
    if username == "not_available":
        raise ValueError("Provide GitHub username.")
    if username == "":
        raise ValueError("Provide GitHub username.")
    if passwd == "":
        raise ValueError("Provide GitHub password/token.")

    upstream_repo_name = args.upstream_repo_name
    upstream_repo_username = args.upstream_repo_username
    your_contribution_directory = args.your_contribution_directory
    cwd = os.getcwd()
    print("Uploading contribution...")

    print("For help: jarvis_upload.py -h\n")

    print("Using GitHub username", username, "\n")
    print("Using GitHub password", passwd, "\n")

    forked_url = "https://github.com/" + username + "/" + upstream_repo_name
    print("Forked_url", forked_url)

    # check if forked url exists
    response = requests.get(forked_url)
    print("response", response)
    print("passwd", passwd)
    # TODO: Use subprocess to enter password
    if response.status_code > 400:
        cmd = (
            "curl -u "
            + username
            + " https://api.github.com/repos/"
            + upstream_repo_username
            + "/"
            + upstream_repo_name
            + "/forks"
            ##+ '--header "Authorization: Bearer '+passwd+'" '
            # + "--header 'authorization: Bearer "+passwd+"' "
            + " -d ''"
            # +"<<"+passwd
        )
        print("Forking repo", cmd)
        os.system(cmd)

    # Takes a few seconds to fork repo
    time.sleep(5)
    print("Note:If you are encoutering issues due to existing forked repo,")
    print("and clashes, delete it and run the script again.")

    if not os.path.exists(upstream_repo_name):
        cmd = "git clone " + forked_url + ".git"
        print("Cloning repo", cmd)
        os.system(cmd)

    if os.path.exists(your_contribution_directory):
        print("Note: adding to existing directory.")
    cmd = (
        "cp -r "
        + your_contribution_directory
        + " jarvis_leaderboard/jarvis_leaderboard/contributions/"
    )
    print("Copying files", cmd)
    os.system(cmd)
    # cmd='cd '+upstream_repo_name
    os.chdir(upstream_repo_name)
    add_dir = "jarvis_leaderboard/contributions/" + your_contribution_directory
    cmd = "ls ./" + add_dir
    print("List files", cmd)
    os.system(cmd)

    print("pwd", os.getcwd())
    cmd = "git add " + add_dir  # + "/*"
    # cmd = "git add ./" + add_dir + "/*"
    print("Git add dir", cmd)
    os.system(cmd)

    # time.sleep(5)
    cmd = "git status "
    # cmd = "git add ./" + add_dir + "/*"
    print("Git status", cmd)
    os.system(cmd)

    cmd = (
        "git commit -m '"
        + "Adding contribution by "
        + str(username)
        + "_"
        + str(your_contribution_directory)
        + ".'"
    )
    print("Git commit", cmd)
    os.system(cmd)

    # cmd = "git remote add origin https://{passwd}@github.com/{username}/jarvis_leaderboard.git"
    # print("Git add origin", cmd)
    # os.system(cmd)

    # cmd = "git remote -v"
    # print("Git -v", cmd)
    # os.system(cmd)

    # cmd="git push https://{passwd}@github.com/{username}/jarvis_leaderboard.git"
    # print("Git push", cmd)
    # os.system(cmd)
    # cmd = "git remote add origin git@github.com:"+username+"/"+"jarvis_leaderboard.git"
    # print("Git add origin", cmd)
    # os.system(cmd)

    # cmd = "git push origin main"
    # cmd = 'git push https://{'+passwd+'}@github.com/{'+username+'}/jarvis_leaderboard.git'
    cmd = (
        "git push https://"
        + passwd
        + "@github.com/"
        + username
        + "/jarvis_leaderboard.git"
    )
    print("Push", cmd)
    os.system(cmd)

    # TODO: add the follwoing insted of python jarvis_leaderboard/rebuild.py
    # errors=rebuild_pages()
    # if len(errors)!=0:
    #   raise ValueError('Found errors in your benchmark, check again',errors)

    cmd = "python jarvis_leaderboard/rebuild.py"
    print(cmd)
    os.system(cmd)

    cmd = (
        "curl -u "
        + username
        + " -d "
        + "'"
        + '{"title":"Adding new contribution by '
        + str(username)
        + "_"
        + str(your_contribution_directory)
        + '","base":"develop", "head":"'
        # + '{"title":"Adding new benchmark","base":"develop", "head":"'
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
