
import re
import ipaddress
from urllib.parse import urlsplit
import tldextract

# Use the same suffix-list configuration as training
extractor = tldextract.TLDExtract(suffix_list_urls=None)

def extract_url_features(url):

    # 1. Normalize input
    url = re.sub(r"\s+", "", str(url).strip())

    if not url:
        raise ValueError("URL cannot be empty")

    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    # 2. Parse URL
    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower()

    if not hostname:
        raise ValueError("Invalid URL: hostname not found")

    # 3. Canonicalize hostname by removing leading www.
    canonical_hostname = re.sub(
        r"^www\.",
        "",
        hostname,
        flags=re.IGNORECASE
    )

    # 4. Extract domain information
    domain_info = extractor(url)

    try:
        ipaddress.ip_address(canonical_hostname)
        is_ip = 1
    except ValueError:
        is_ip = 0

    domain_length = len(canonical_hostname)
    tld_length = len(domain_info.suffix)

    subdomains = [
        part for part in domain_info.subdomain.split(".")
        if part and part.lower() != "www"
    ]

    no_of_subdomain = len(subdomains)

    # 5. Create canonical URL text:
    # no scheme, no leading www.
    canonical_url = (
        canonical_hostname
        + (parsed.path or "")
        + (("?" + parsed.query) if parsed.query else "")
        + (("#" + parsed.fragment) if parsed.fragment else "")
    )

    # 6. Character statistics
    url_length = len(canonical_url)

    letters = sum(c.isalpha() for c in canonical_url)
    digits = sum(c.isdigit() for c in canonical_url)

    equals = canonical_url.count("=")
    question_marks = canonical_url.count("?")
    ampersands = canonical_url.count("&")

    other_special = sum(
        not c.isalnum() and c not in "=?&"
        for c in canonical_url
    )

    # 7. Obfuscation indicators
    encoded_sequences = re.findall(
        r"%[0-9a-fA-F]{2}",
        canonical_url
    )

    obfuscated_chars = len(encoded_sequences)

    has_obfuscation = int(
        obfuscated_chars > 0 or "@" in parsed.netloc
    )

    # 8. Ratios
    denominator = max(url_length, 1)

    return {
        "URLLength": url_length,
        "DomainLength": domain_length,
        "IsDomainIP": is_ip,
        "TLDLength": tld_length,
        "NoOfSubDomain": no_of_subdomain,
        "HasObfuscation": has_obfuscation,
        "NoOfObfuscatedChar": obfuscated_chars,
        "ObfuscationRatio": obfuscated_chars / denominator,
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": letters / denominator,
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": digits / denominator,
        "NoOfEqualsInURL": equals,
        "NoOfQMarkInURL": question_marks,
        "NoOfAmpersandInURL": ampersands,
        "NoOfOtherSpecialCharsInURL": other_special,
        "SpacialCharRatioInURL": other_special / denominator
    }


FEATURE_NAMES = ['URLLength', 'DomainLength', 'IsDomainIP', 'TLDLength', 'NoOfSubDomain', 'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio', 'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL', 'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL']
