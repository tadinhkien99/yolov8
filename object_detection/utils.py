#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    utils.py
# @Author:      kien.tadinh
# @Time:        3/16/2024 9:34 AM

import os


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def server_connect(server_ip, server_port, server_username, server_password):
    pass


def server_recheck(server_recheck):
    pass
