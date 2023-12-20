#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <github_username>"
  exit 1
fi

USERNAME=$1
REPO="datamove/linux-git2"
PER_PAGE=100

PULLS_COUNT=0
EARLIEST_PR=0
MERGED_STATUS=0
PAGE=1

while true; do
  PR_INFO=$(curl -s "https://api.github.com/repos/$REPO/pulls?state=all&user=$USERNAME&per_page=$PER_PAGE&page=$PAGE")

  CURRENT_PULLS_COUNT=$(echo "$PR_INFO" | jq '. | length')
  if [ "$CURRENT_PULLS_COUNT" -eq 0 ]; then
    break
  fi

  for ((i = 0; i < CURRENT_PULLS_COUNT; i++)); do
    CURRENT_PR_NUMBER=$(echo "$PR_INFO" | jq -r ".[$i].number")
    CURRENT_PR_USER=$(echo "$PR_INFO" | jq -r ".[$i].user.login")

    if [ "$CURRENT_PR_USER" = "$USERNAME" ]; then
      PULLS_COUNT=$((PULLS_COUNT + 1))
      if [ "$EARLIEST_PR" -eq 0 ] || [ "$CURRENT_PR_NUMBER" -lt "$EARLIEST_PR" ]; then
        EARLIEST_PR=$CURRENT_PR_NUMBER
      fi
    fi
  done

  PAGE=$((PAGE + 1))
done

if [ "$PULLS_COUNT" -eq 0 ]; then
  echo "PULLS 0"
  echo "EARLIEST N/A"
  echo "MERGED 0"
  exit 0
fi

MERGED_STATUS=$(curl -s "https://api.github.com/repos/$REPO/pulls/$EARLIEST_PR" | grep -E '"merged": true')

echo "PULLS $PULLS_COUNT"
echo "EARLIEST $EARLIEST_PR"
echo "MERGED $([ -n "$MERGED_STATUS" ] && echo 1 || echo 0)"

