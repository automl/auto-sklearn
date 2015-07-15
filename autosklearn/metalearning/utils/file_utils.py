import StringIO
import tempfile
import urllib2

import xmltodict

try:
    from lxml import etree
except:
    pass


def xml_to_dict(xml_string, schema_string=None):
    # _validate_xml_against_schema(xml_string, schema_string)
    xml_dict = xmltodict.parse(xml_string)
    return xml_dict


def _validate_xml_against_schema(xml_string, schema_string):
    """Starting point for this code from http://stackoverflow.com/questions/
    17819884/xml-xsd-feed-validation-against-a-schema"""
    if type(schema_string) == str:
        schema = etree.XML(schema_string)
    elif type(schema_string) == etree.XMLSchema:
        schema = schema_string
    else:
        schema = etree.XMLSchema(schema_string)
    xmlparser = etree.XMLParser(schema=schema)
    etree.fromstring(xml_string, xmlparser)

def read_url(url):
    CHUNK = 16 * 1024
    string = StringIO.StringIO()
    connection = urllib2.urlopen(url)
    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
    with tmp as fh:
        while True:
            chunk = connection.read(CHUNK)
            if not chunk: break
            fh.write(chunk)

    tmp = open(tmp.name, "r")
    with tmp as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk: break
            string.write(chunk)

    return string.getvalue()