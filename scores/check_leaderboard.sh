#!/bin/bash

# This script should be executed two times per day.
#
# $ sudo crontab -e
# 0 4 * * * /media/spotify2018/spotify2018/Spotify\ new\ by\ lele/Spotify-Challenge/scores/check_leaderboard.sh
# 0 16 * * * /media/spotify2018/spotify2018/Spotify\ new\ by\ lele/Spotify-Challenge/scores/check_leaderboard.sh
#
# Executes these commands in background.
# 1 - Leaderboard scraping
# 2 - Git commit score folder
# 3 - Git push

/home/spotify2018/anaconda3/bin/python3.6 /media/spotify2018/spotify2018/Spotify\ new\ by\ lele/Spotify-Challenge/scores/scraping.py && /usr/bin/git commit -m 'Leaderboard updated by Keplero' -- /media/spotify2018/spotify2018/Spotify\ new\ by\ lele/Spotify-Challenge/scores/ && /usr/bin/git push
