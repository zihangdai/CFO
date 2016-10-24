# -*- coding: utf-8 -*-
import json
import urllib
import re

api_key = 'AIzaSyAsmMIiVDkF2Vfjt3cDwSHCmHF7QTS0_kY'

def suggest_id(query_string):
    service_url = 'https://www.googleapis.com/freebase/v1/search'
    params = {
        'query': query_string,
        'key': api_key
    }
    url = service_url + '?' + urllib.urlencode(params)
    response = json.loads(urllib.urlopen(url).read())

    suggested_entity = []
    for result in response['result']:
        if result['mid'].startswith('/m/'):
            suggested_entity.append('fb:m.' + str(result['mid'].split('/m/')[-1]))

    return suggested_entity

def mid2name(entity_mid):
    service_url = 'https://www.googleapis.com/freebase/v1/mqlread'
    query = [{'id': None, 'mid': entity_mid, 'name': None}]

    params = {
            'query': json.dumps(query),
            'key': api_key
    }

    url = service_url + '?' + urllib.urlencode(params)
    response = json.loads(urllib.urlopen(url).read())

    if response['result'][0].has_key('name') and response['result'][0]['name']:
        return response['result'][0]['name'].encode('utf-8')
    else:
        return None

def mid2id(entity_mid):
    service_url = 'https://www.googleapis.com/freebase/v1/mqlread'
    query = [{'mid': entity_mid, 'id': None}]

    params = {
            'query': json.dumps(query),
            'key': api_key
    }

    url = service_url + '?' + urllib.urlencode(params)
    response = json.loads(urllib.urlopen(url).read())

    if response['result'][0].has_key('id'):
        return response['result'][0]['id']
    else:
        return None


def id2mid(entity_id):
    service_url = 'https://www.googleapis.com/freebase/v1/mqlread'
    query = [{'id': entity_id, 'mid': None}]

    params = {
            'query': json.dumps(query),
            'key': api_key
    }

    url = service_url + '?' + urllib.urlencode(params)
    response = json.loads(urllib.urlopen(url).read())

    if response['result'][0].has_key('mid'):
        return response['result'][0]['mid']
    else:
        return None

def name2mids(entity_name):
    service_url = 'https://www.googleapis.com/freebase/v1/mqlread'
    query = [{'name': entity_name, 'mid': None, '/common/topic/alias': []}]

    params = {
            'query': json.dumps(query),
            'key': api_key
    }

    url = service_url + '?' + urllib.urlencode(params)
    response = json.loads(urllib.urlopen(url).read())

    mid_list = []
    for res in response['result']:
        if res.has_key('mid'):
            mid_list.append(str(res['mid']))
    return mid_list

def unquotekey(key, encoding=None):
    """
    unquote a namespace key and turn it into a unicode string
    """

    valid_always = string.ascii_letters + string.digits

    output = []
    i = 0
    while i < len(key):
        if key[i] in valid_always:
            output.append(key[i])
            i += 1
        elif key[i] in '_-' and i != 0 and i != len(key):
            output.append(key[i])
            i += 1
        elif key[i] == '$' and i+4 < len(key):
            # may raise ValueError if there are invalid characters
            output.append(unichr(int(key[i+1:i+5],16)))
            i += 5
        else:
            raise ValueError, "unquote key saw invalid character '%s' at position %d" % (key[i], i)

    ustr = u''.join(output)
    
    if encoding is None:
        return ustr

    return ustr.encode(encoding)

# used to escape strings for sparql query
def escape_string(s):
    escape_map = {
        '"' : '\\"',
        '\r': '\\r',
        '\n': '\\n',
        '\t': '\\t',
        '\b': '\\b',
        '\f': '\\f'
    }
    s = s.replace('\\','\\u005c\\u005c')
    for key, value in escape_map.items():
        s = s.replace(key,value)
    return '"' + s + '"'

# used to escape strings for sparql query
def unescape_string(s):
    unescape_map = {
        '\\"': '"' ,
        '\\r': '\r',
        '\\n': '\n',
        '\\t': '\t',
        '\\b': '\b',
        '\\f': '\f'
    }
    # strip the quote " on both sides
    s = s[1:-1]
    for key, value in unescape_map.items():
        s = s.replace(key,value)
    s = s.replace('\\u005c\\u005c', '\\')
    return s


