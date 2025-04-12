#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Miao at 2025-04-04
'''
@File   : check_files
@Time   : 2025-04-04 2:47 p.m.
@Author : Miao
@Project: IFT6289-multimodal-rag 
@Desc   : Please enter here
'''
import os
def is_image_file(filename):
    '''
    Return true if filename ends with a known image file.
    :param filename: The image filename with path
    :type filename: str
    :return: True/False
    :rtype: Boolean
    '''
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    ext = os.path.splitext(filename.lower())[1]
    return ext in image_extensions