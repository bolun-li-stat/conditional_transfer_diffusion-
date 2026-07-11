# ADM integration history

The earlier ADM change was merged into its feature base rather than the default branch. The default branch therefore contained the evaluation foundation but not the ADM files even though the earlier change appeared merged in the hosting interface.

This branch starts from the latest default-branch commit `5add8ad`. Its merge-base with the default branch is that same commit. It integrates only the six ordinary ADM commits from the range after `cfcb0f2` through `043ef63`; it does not cherry-pick merge commit `7ab630f` and does not duplicate the evaluation commits already present.

Before review, verify with:

```bash
git merge-base origin/main HEAD
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
```

The final PR records the exact base and head SHAs after all readiness changes are committed.
