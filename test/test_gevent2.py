# -*- coding:utf-8 -*-
import gevent
import socket
import time
from gevent import monkey
# urls = ['www.baidu.com', 'www.gevent.org', 'www.python.org']
# jobs = [gevent.spawn(socket.gethostbyname, url) for url in urls]
# gevent.joinall(jobs, timeout=5)
# print([job.value for job in jobs])

def fake_get(url):
    time.sleep(2)
    return url

monkey.patch_all()
a = time.time()
urls = ['www.baidu.com', 'www.gevent.org', 'www.python.org']
jobs = [gevent.spawn(fake_get, url) for url in urls]
gevent.joinall(jobs, timeout=5)
print([job.value for job in jobs])
print(time.time() - a)
