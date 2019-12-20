#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:yaoli 
@file: utils.py 
@time: 2019/12/19 
"""
import progressbar


def get_progress_bar(total):
    format_custom_text = progressbar.FormatCustomText(
        'Loss: %(loss).3f | Acc: %(acc).3f%% (%(c)d/%(t)d)',
        dict(
            loss=0,
            acc=0,
            c=0,
            t=0,
        ),
    )
    prog_bar = progressbar.ProgressBar(0, total, widgets=[
        progressbar.Counter(), ' of {} '.format(total),
        progressbar.Bar(),
        ' ', progressbar.ETA(),
        ' ', format_custom_text
    ])
    return prog_bar, format_custom_text


def update_progress_bar(progress_bar_obj, index=None, loss=None, acc=None, c=None, t=None, ):
    prog_bar, format_custom_text = progress_bar_obj
    format_custom_text.update_mapping(loss=loss, acc=acc, c=c, t=t)
    prog_bar.update(index)


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
