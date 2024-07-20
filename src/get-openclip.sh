#!/usr/bin/env bash

BRANCHNAME="main"
COMMITID="ee286275771f4efccdd5ac6df63ce4233c7d9ce8"
git clone -b $BRANCHNAME --recursive https://github.com/mlfoundations/open_clip && cd open_clip && git checkout -b $BRANCHNAME $COMMITID 
