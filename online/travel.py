#!/usr/bin/env python
from datetime import datetime, timedelta
import httplib2
from lxml import etree
import sys
import time
from urllib import urlencode

API_KEY = "api_key_goes_here"

class Kayak(object):

    http = httplib2.Http()
    url_prefix = 'http://api.kayak.com'
    session = None
    session_created = None
    session_expiry = timedelta(minutes=30)
    cookie = None
    user_agent = 'Iox-kayak-search'

    def __init__(self, key):
        self.key = key

    def set_session(self):
        params = [('token', self.key)]
        url = '%s/k/ident/apisession?%s' % (self.url_prefix, urlencode(params))
        response, content = self.http.request(url)
        xml = etree.fromstring(content)
        error = xml.find('error')
        if error.text:
            raise Exception(error.text)
        self.session = xml.find('sid').text
        self.session_created =  datetime.now()
        self.cookie = response['set-cookie']

    def get_session(self):
        now = datetime.now()
        if not self.session or (now - self.session_created) > self.session_expiry:
            self.set_session()
        return self.session

    def start_flight_search(self, origin, destination, depart_date, return_date, \
            depart_time='a', return_time='a', travelers=2, cabin='e', oneway=False):

        params = [
            ('action', 'doflights'),
            ('apimode', '1'),
            ('basicmode', 'true'),
            ('oneway', 'y' if oneway else 'n'),
            ('origin', origin),
            ('destination', destination),
            ('depart_date', depart_date),
            ('return_date', return_date),
            ('depart_time', depart_time),
            ('return_time', return_time),
            ('travelers', travelers),
            ('cabin', cabin),
            ('_sid_', self.get_session()),
        ]
        url = '%s/s/apisearch?%s' % (self.url_prefix, urlencode(params))
        headers = {'Cookie': self.cookie, 'User-Agent': self.user_agent}
        response, content = self.http.request(url, headers=headers)

        xml = etree.fromstring(content)
        search_id = xml.find('searchid')
        if search_id is None or search_id.text == '':
            raise Exception('Failed to start the search')

        return search_id.text

    def get_flight_results(self, search_id, count=10):
        params = [
            ('_sid_', self.get_session()),
            ('c', count),
            ('searchid', search_id),
            ('apimode', '1'),
        ]
        url = '%s/s/apibasic/flight?%s' % (self.url_prefix, urlencode(params))
        headers = {'Cookie': self.cookie, 'User-Agent': self.user_agent}
        response, content = self.http.request(url, headers=headers)
        return etree.fromstring(content)

    def find_flights(self, origin, destination, depart_date, return_date, \
            depart_time='a', return_time='a', travelers=2, cabin='e', oneway=False, count=10):

        search_id = self.start_flight_search(origin, destination, depart_date, return_date, \
            depart_time, return_time, travelers, cabin, oneway)

        sys.stdout.write('Getting flights')
        while True:
            results = self.get_flight_results(search_id, count)
            more = results.find('morepending')
            if more.text is None or more.text.lower() != 'true':
                break
            for i in range(10):
                sys.stdout.write('.')
                sys.stdout.flush()
                time.sleep(1)

        print(results.find('count').text)


if __name__ == '__main__':
    k = Kayak(API_KEY)
    k.find_flights('MSP', 'HNL', '01/23/2010', '01/30/2010')
