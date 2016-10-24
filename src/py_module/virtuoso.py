#!/usr/bin/python
import sys
import urllib, json
import freebase

# Setting global variables
data_source = 'fb:'
query_url   = 'http://localhost:8890/sparql/'

# HTTP URL is constructed accordingly with JSON query results format in mind.
def sparql_query(query, URL, format='application/json'):

    params={
             'default-graph': '',
             'should-sponge': 'soft',
             'query': query.encode('utf8'),
             'debug': 'on',
             'timeout': '',
             'format': format,
             'save': 'display',
        'fname': ''
    }

    encoded_query = urllib.urlencode(params)
    http_response = urllib.urlopen(URL, encoded_query).read()

    try:
        json_response = json.loads(http_response)
        return json_response
    except:
        print >> sys.stderr, 'json load error'
        print >> sys.stderr, http_response
        return None 

# Using freebase mid to query its types
def id_query_type(node_id):
    query = '''
        SELECT ?type WHERE {<%s> <fb:type.object.type> ?type}
    ''' % (node_id)
    json_response = sparql_query(query, query_url)

    try:
        type_list = [item['type']['value'] for item in json_response['results']['bindings']]
        return list(set(type_list))
    except:
        return []

# Using freebase mid to query its original cased name
def id_query_en_name(node_id):
    query = '''
        SELECT ?name WHERE {<%s> <fb:type.object.en_name> ?name}
    ''' % (node_id)
    json_response = sparql_query(query, query_url)

    try:
        name_list = [item['name']['value'] for item in json_response['results']['bindings']]
        return list(set(name_list))
    except:
        return []

# Using freebase mid to query its original cased alias
def id_query_en_alias(node_id):
    query = '''
        SELECT ?alias WHERE {<%s> <fb:common.topic.en_alias> ?alias}
    ''' % (node_id)
    json_response = sparql_query(query, query_url)

    try:
        alias_list = [item['alias']['value'] for item in json_response['results']['bindings']]
        return list(set(alias_list))
    except:
        return []

# Using freebase mid to query its processed & tokenized name
def id_query_name(node_id):
    query = '''
        SELECT ?name WHERE {<%s> <fb:type.object.name> ?name}
    ''' % (node_id)
    json_response = sparql_query(query, query_url)

    try:
        name_list = [item['name']['value'] for item in json_response['results']['bindings']]
        return list(set(name_list))
    except:
        return []

# Using freebase mid to query its processed & tokenized alias
def id_query_alias(node_id):
    query = '''
        SELECT ?alias WHERE {<%s> <fb:common.topic.alias> ?alias}
    ''' % (node_id)
    json_response = sparql_query(query, query_url)

    try:
        alias_list = [item['alias']['value'] for item in json_response['results']['bindings']]
        return list(set(alias_list))
    except:
        return []

# Using freebase mid to query its processed & tokenized name & alias
def id_query_str(node_id):
    query = '''
        SELECT ?str WHERE { {<%s> <fb:type.object.name> ?str} UNION {<%s> <fb:common.topic.alias> ?str} }
    ''' % (node_id, node_id)
    json_response = sparql_query(query, query_url)

    try:
        name_list = [item['str']['value'] for item in json_response['results']['bindings']]
        return list(set(name_list))
    except:
        return []
# Using freebase mid to query all relations coming out of the entity
def id_query_out_rel(node_id, unique = True):
    query = '''
        SELECT ?relation WHERE {<%s> ?relation ?object}
    ''' % (node_id)
    json_response = sparql_query(query, query_url)

    try:
        relations = [str(item['relation']['value']) for item in json_response['results']['bindings']] 
        return list(set(relations))
    except:
        return []

# Using freebase mid to query all relations coming into the entity
def id_query_in_rel(node_id, unique = True):
    query = '''
        SELECT ?relation WHERE {?subject ?relation <%s>}
    ''' % (node_id)
    json_response = sparql_query(query, query_url)

    try:
        relations = [str(item['relation']['value']) for item in json_response['results']['bindings']] 
        return list(set(relations))
    except:
        return []


# Using the name of an entity to query its freebase mid
def name_query_id(name):
    query = '''
        SELECT ?node_id WHERE {?node_id <fb:type.object.name> "%s"}
    ''' % (name)
    json_response = sparql_query(query, query_url)

    try:
        node_id_list = [str(item['node_id']['value']) for item in json_response['results']['bindings']]
        return list(set(node_id_list))
    except:
        return []

# Using the alias of an entity to query its freebase mid
def alias_query_id(alias):
    query = '''
        SELECT ?node_id WHERE {?node_id <fb:common.topic.alias> "%s"}
    ''' % (alias)
    json_response = sparql_query(query, query_url)

    try:
        node_id_list = [str(item['node_id']['value']) for item in json_response['results']['bindings']]
        return list(set(node_id_list))
    except:
        return []

# Using the alias/name of an entity to query its freebase mid
def str_query_id(string):
    query = '''
        SELECT ?node_id WHERE  { {?node_id <fb:common.topic.alias> "%s"} UNION {?node_id <fb:type.object.name> "%s"} }
    ''' % (string, string)
    json_response = sparql_query(query, query_url)

    try:
        node_id_list = [str(item['node_id']['value']) for item in json_response['results']['bindings']]
        return list(set(node_id_list))
    except:
        return []

# Using freebase mid to query all object coming out of the entity
def id_query_in_entity(node_id, unique = True):
    query = '''
        SELECT ?subject WHERE {?subject ?relation <%s>}
    ''' % (node_id)
    json_response = sparql_query(query, query_url)

    try:
        subjects = [str(item['subject']['value']) for item in json_response['results']['bindings']] 
        return list(set(subjects))
    except:
        return []

# Using freebase mid to query all relation coming into the entity
def id_query_out_entity(node_id, unique = True):
    query = '''
        SELECT ?object WHERE {<%s> ?relation ?object}
    ''' % (node_id)
    json_response = sparql_query(query, query_url)

    try:
        objects = [str(item['object']['value']) for item in json_response['results']['bindings']] 
        return list(set(objects))
    except:
        return []

# Using the subject and relation to query the corresponding object
def query_object(subject, relation):
    query = '''
        SELECT ?object WHERE {<%s> <%s> ?object}
    ''' % (subject, relation)
    json_response = sparql_query(query, query_url)

    try:
        return [str(item['object']['value']) for item in json_response['results']['bindings']]
    except:
        return []

# Using the object and relation to query the corresponding subject 
def query_subject(obj, relation):
    query = '''
        SELECT ?subject WHERE {?subject <%s> <%s>}
    ''' % (relation, obj)
    json_response = sparql_query(query, query_url)

    try:
        return [str(item['subject']['value']) for item in json_response['results']['bindings']]
    except:
        return []

# Using the subject and object to query the corresponding relation
def query_relation(sub, obj):
    query = '''
        SELECT ?relation WHERE {<%s> ?relation <%s>}
    ''' % (sub, obj)
    json_response = sparql_query(query, query_url)

    try:
        objects = [str(item['relation']['value']) for item in json_response['results']['bindings']] 
        return list(set(objects))
    except:
        return []

# Check whether a node is a CVT node
def check_cvt(node_id):
    query = '''
        SELECT ?tag WHERE {<%s> <fb:cvt_node_identifier> ?tag}
    ''' % (node_id)
    json_response = sparql_query(query, query_url)
    ret = [str(item['tag']['value']) for item in json_response['results']['bindings']]

    if len(ret) == 1 and ret[0] == 'true':
        return True
    else:
        return False
