import math
import numpy as np
import logging
import requests
import tldextract
import datetime
from datetime import datetime
from collections import Counter

# def time(timestamp): # Timestamp in dataset
#     """ Returns a string represents the time in terms of minutes and seconds
#     Parameters:
#         timestamp: timestamp which will be converted to time object
#     """
#     return datetime.fromtimestamp(int(timestamp)/1000).strftime('%M:%S.%f') 

def count_digits(domain): # numeric in dataset
    """ Returns an integer represents the number of digits in domain
    Parameters:
        domain: the domain name received in logs
    """
    count = 0
    for c in domain:
        if c.isdigit():
            count += 1
    return count

def count_char(domain): # FQDN in dataset
    count = 0
    for c in domain:
        if c == '.': # removing the dot from the count
            continue
        else:
            count += 1
    return count

def count_labels(domain): # labels in dataset
    """Returns an integer represents the number of the labels
    Parameters:
        domain: the domain name received in logs
    Example: 
        "www.scholar.google.com", there are four labels separated by dots
    """
    count = 0 
    for c in domain:
        if c == '.': # removing the dot from the count
            count += 1
    return count + 1

def count_lower(domain):  # lower in dataset
    """Returns an integer represents the number of the lowercase letters
    Parameters:
        domain: the domain name received in logs
    """
    count = 0
    for c in domain:
        if (c.islower()):
            count += 1
    return count

def count_upper(domain): # upper in dataset
    """Returns an integer represents the number of the uppercase letters
    Parameters:
        domain: the domain name received in logs
    """
    count = 0
    for c in domain:
        if (c.isupper()):
            count += 1
    return count

def entropy(domain): # entropy in dataset
    """ Returns the metric entropy (Shannon's entropy divided by string length)
    Based on: https://kldavenport.com/detecting-randomly-generated-domains/
    """
    # Counter counts hashable objects
    counter, length = Counter(domain), np.float(len(domain)) # count hashable objects and the lens
    return -np.sum( val/length * np.log2(val/length) for val in counter.values()) # calculating the entropy

def length(domain): # Len in dataset
    """Returns an integer represents the number of the total letters in "domain and subdomain" only
    Parameters:
        domain: the domain name received in logs
    """
    ex = tldextract.extract(domain)

    return len(ex.domain) + len(ex.subdomain)

def get_labels(domain): # getting a list of the labels
    """ Returns a list of the labels exist in the domain
    Parameters:
        domain: the domain name received in logs
    """
    return domain.split(sep = '.')

def labels_max(domain): # labels_max in dataset
    """ Returns maximum number of letters labels exist in the domain
    Parameters:
        domain: the domain name received in logs
    """
    labels = get_labels(domain)
    num_words = len(labels)
    
    maximum = 0
    for word in labels:
        length = len(word)
        if length > maximum:
            maximum = length
    return maximum

def labels_average(domain):# labels_average in dataset
    """Returns an integer represents the number of letters in the longest word
    Parameters:
        domain: the domain name received in logs
    """
    labels = get_labels(domain)
    num_words = len(labels)
    
    if num_words:
        total = 0
        for word in labels:
            total += len(word)
    else: return 0
    
    return total / num_words

def longest_word(domain): # longest_word in dataset
    """Returns an integer represents the number of letters in the longest word
    Parameters:
        domain: the domain name received in logs
    """
    labels = get_labels(domain)
    maximum = 0
    for word in labels:
        length = len(word)
        if length > maximum:
            maximum = length
    return maximum

def contain_subdomain(domain): # subdomain in dataset
    """Returns 1 if there is a subdomain and 0 otherwise
    Parameters:
        domain: the domain name received in logs
    """
    ex = tldextract.extract(domain)
    if ex.subdomain:
        return 1
    else:
        return 0
    
def subdomain_length(domain): # subdomain_length in dataset
    """Returns an integer represents the number of chars in the subdomain
    Parameters:
        domain: the domain name received in logs
    """
    if contain_subdomain(domain):
        count = 0
        ex = tldextract.extract(domain)
        for c in ex.subdomain:
            if c == '.':
                continue
            else: count += 1
        return count
    else: return 0
    
def count_special_character(domain):
    """Returns an integer represents the number of special chars in domain
    Parameters:
        domain: the domain name received in logs
    """
    count = 0
    for c in domain:
        if (32 <= ord(c) <= 47) or (58 <= ord(c) <= 64) or (91 <= ord(c) <= 96) or (123 <= ord(c) <= 126):
            count += 1
    return count

def second_level_domain(domain): # SLD in dataset
    """Returns an string represents the subdomain
    Parameters:
        domain: the domain name received in logs
    """
    return tldextract.extract(domain).domain













