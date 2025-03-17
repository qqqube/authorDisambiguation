import configparser
import time
from selenium import webdriver
import openreview
import requests
from pdf2image import convert_from_path
import http.client
import json
from openai import OpenAI
import base64
import numpy as np


# ------ Author and Paper Objects/Methods ----------

def get_preferred_name(name_lst):
    """
    Given a list of AuthorNameOR objects (contains at least one item).
    return the preferred name
    """
    for name in name_lst:
        if name.preferred == True:
            return name
    # at this point, name_lst must have more than one AuthorNameOR object
    # that contains names that either have preferred set to False or None
    # We would prefer a name that has preferred set to False over one that's set to None
    name_lst = sorted(name_lst, key=lambda name: (name.fullname, name.username)) # just to make function deterministic
    preferred_name_lst = [name for name in name_lst if name.preferred is not None]
    no_preferred_name_lst = [name for name in name_lst if name.preferred is None]
    if len(preferred_name_lst) > 0:
        return preferred_name_lst[0]
    return no_preferred_name_lst[0]

def sigmoid(x):
    return 1 / (1 + (np.e ** (-x)))

class AuthorNameOR:
    def __init__(self, _dict):
        self.fullname = _dict["fullname"]
        self.preferred = _dict["preferred"] # True, False, or None
        self.username = _dict["username"]

class AuthorHistoryOR:
    def __init__(self, _dict):
        self.position = _dict["position"] # string
        self.start = _dict["start"] # year
        self.end = _dict["end"] # year (could be None)
        self.institution = _dict["institution"] # string

class AuthorPersonalLinkOR:
    def __init__(self, _dict):
        self.homepage = _dict["homepage"]
        self.gscholar = _dict["gscholar"]
        self.dblp = _dict["dblp"]
        self.orcid = _dict["orcid"]
        self.linkedin = _dict["linkedin"]
        self.semantic_scholar = _dict["semanticScholar"]
        self.aclanthology = _dict["aclanthology"]

class AuthorOpenReviewProfile:
    def __init__(self, _dict):
        self.names = [AuthorNameOR(item) for item in _dict["names"]]
        self.history = [AuthorHistoryOR(item) for item in _dict["career_edu_history"]]
        self.personal_links = AuthorPersonalLinkOR(_dict["personal_links"])
        self.fullname, self.preferred_author_id = self._get_preferred_name()
    def _get_preferred_name(self):
        preferred_name = get_preferred_name(self.names)
        return preferred_name.fullname, preferred_name.username
    def get_homepage(self):
        return self.personal_links.homepage
    def get_gscholar(self):
        return self.personal_links.gscholar
    def get_affiliation(self):
        if len(self.history) > 0:
            return self.history[0].institution
        return None
    def get_or_url(self):
        return f"https://openreview.net/profile?id={self.preferred_author_id}"

class WebSearchItem():
    def __init__(self, url, affiliation):
        self.url = url
        self.affiliation = affiliation # could be None

class WebSearchURLs():
    def __init__(self, author_name):
        """At least one of the URLs must not be None"""
        self.homepage = None
        self.gscholar = None
        self.author_name = author_name
    def set_homepage_url(self, url, affiliation):
        self.homepage = WebSearchItem(url, affiliation)
    def set_gscholar_url(self, url, affiliation):
        self.gscholar = WebSearchItem(url, affiliation)
    def merge(self, websearch_urls):
        if self.homepage is None:
            self.homepage = websearch_urls.homepage
        if self.gscholar is None:
            self.gscholar = websearch_urls.gscholar
    def has_urls(self):
        return (self.homepage is not None) or (self.gscholar is not None)


class Author:
    def __init__(self, name_lst=None, openreview_profile=None,
                 websearch_urls=None):
        self.papers = []
        self.openreview_profile, self.name_lst = None, None
        self.websearch_urls = None
        self.merge_authors = [] # authors without URLs
        if openreview_profile:
            self.openreview_profile = openreview_profile
            self.fullname = self.openreview_profile.fullname
        elif name_lst:
            self.name_lst = name_lst
            self.fullname = get_preferred_name(self.name_lst).fullname
        elif websearch_urls:
            if websearch_urls.has_urls():
                self.websearch_urls = websearch_urls
                self.fullname = websearch_urls.author_name
            else:
                self.fullname = websearch_urls.author_name
    def merge_author(self, author_inst, intersection_size):
        author_inst.websearch_urls = self.websearch_urls
        author_inst.openreview_profile = self.openreview_profile
        self.merge_authors.append((intersection_size, author_inst))
    def get_coauthors(self):
        coauthors = []
        for paper in self.papers:
            coauthors += [author for author in paper.author_instances if author != self]
        return list(set(coauthors))
    def has_urls(self):
        return (self.homepage_url() is not None) or (self.gscholar_url() is not None) or (self.has_openreview_profile())
    def add_paper(self, paper):
        self.papers.append(paper)
    def add_websearch_urls(self, websearch_urls):
        if self.websearch_urls is None:
            self.websearch_urls = websearch_urls
        else:
            self.websearch_urls.merge(websearch_urls)
    def confidence_score(self, intersection_size):
        """Return confidence score between 0 and 1 based on intersection_size"""
        assert intersection_size > 0
        # intersection_size is an integer greater than 0
        return sigmoid(intersection_size)
    def get_papers(self):
        """Return papers sorted by number of affiliations (confident predictions)"""
        return sorted(self.papers, key=lambda paper: len(paper.institution_names))
    def get_merged_papers(self):
        lst = []
        for intersection_size, merged_author_inst in self.merge_authors:
            papers = merged_author_inst.get_papers()
            for paper in papers:
                lst.append((paper.pdf_name(), self.confidence_score(intersection_size)))
        return lst
    def has_openreview_profile(self):
        return self.openreview_profile is not None
    def openreview_url(self):
        if self.has_openreview_profile():
            return self.openreview_profile.get_or_url()
        return None
    def homepage_url(self):
        homepages = []
        if self.openreview_profile:
            homepages.append(self.openreview_profile.get_homepage())
        if self.websearch_urls and self.websearch_urls.homepage:
            homepages.append(self.websearch_urls.homepage.url)
        homepages = [item for item in homepages if item is not None]
        if len(homepages) > 0:
            # prefer to return openreview homepage
            return homepages[0]
        return None
    def gscholar_url(self):
        gscholar_urls = []
        if self.openreview_profile:
            gscholar_urls.append(self.openreview_profile.get_gscholar())
        if self.websearch_urls and self.websearch_urls.gscholar:
            gscholar_urls.append(self.websearch_urls.gscholar.url)
        gscholar_urls = [item for item in gscholar_urls if item is not None]
        if len(gscholar_urls) > 0:
            return gscholar_urls[0]
        return None
    def affiliation(self):
        if self.openreview_profile:
            or_affiliation = self.openreview_profile.get_affiliation()
            if or_affiliation is not None:
                return or_affiliation
        if self.websearch_urls:
            if self.websearch_urls.homepage:
                affiliation = self.websearch_urls.homepage.affiliation
                if affiliation is not None:
                    return affiliation
            if self.websearch_urls.gscholar:
                affiliation = self.websearch_urls.gscholar.affiliation
                if affiliation is not None:
                    return affiliation
        return None


class Paper:
    def __init__(self, _dict):
        self.id = _dict["id"]
        self.title = _dict["title"]
        self.author_names = _dict["author_names"]
        self.institution_names = _dict["institution_names"]
        self.openreview_url = _dict["openreview_url"]
        self.author_instances = []
    def hosted_on_openreview(self):
        return self.openreview_url is not None
    def add_author(self, author_inst):
        self.author_instances.append(author_inst)
    def pdf_name(self):
        return f"{self.id}.pdf"


# ------ Login Helper Functions -------------------------------------

def get_credentials(credentials_path):
    """
    Credentials INI file should look like:

    [BASIC]
    USERNAME = <username>
    PASSWORD = <password>
    SERPER_API_KEY = <api_key>
    OPENAI_API_KEY = <api_key>
    """
    config = configparser.ConfigParser()
    config.read(credentials_path)
    return config["BASIC"]["USERNAME"], config["BASIC"]["PASSWORD"], config["BASIC"]["SERPER_API_KEY"], config["BASIC"]["OPENAI_API_KEY"]

# -------- Serper Search Engine API -------------------------------------

class SerperResult():

    def __init__(self, _dict):
        self.title = _dict["title"]
        self.link = _dict["link"]
        self.snippet = _dict["snippet"]
        self.position = _dict["position"]
    
    def __str__(self):
        return f"position: {self.position} \ntitle: {self.title} \nlink: {self.link}\nsnippet: {self.snippet}"


class Serper():

    def __init__(self, api_key):
        self.api_key = api_key
    
    def query(self, _query):
        """ Return list of SerperResult instances sorted by position """
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({"q": _query})
        headers = {'X-API-KEY': self.api_key, 'Content-Type': 'application/json'}
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = res.read()
        results = json.loads(data.decode("utf-8"))
        results = [SerperResult(item) for item in results['organic']]
        results.sort(key=lambda item : item.position)
        return results


# ------- OpenAI Helper Functions --------------------------
# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def gpt4o(client, prompt, base64_image):
    print(f"gpt4o: {prompt}")
    response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
        )
    output = response.choices[0].message.content
    print(f"gpt4o output: {output}")
    return output


# ------- Save PDFs Locally and Take Picture of First Page ------------------

def save_pdf_from_url(url, output_path):
    """ Saves PDF locally """
    response = requests.get(url)
    with open(output_path, 'wb') as f:
        f.write(response.content)

def save_first_pdf_page_as_img(path_to_pdf, output_path):
    """ Converts first page of local pdf into image """
    img_list = convert_from_path(path_to_pdf, dpi=800, first_page=1, last_page=1)
    img_list[0].save(output_path, "JPEG")


# ----- Selenium Helper Functions ------------------------------------------

def capture_webpage(url, save_path, zoom_out=True):
    """
    Take screenshot of webpage at provided url and save to save_path (local)
    """
    driver = webdriver.Chrome()
    driver.get(url)

    time.sleep(10)

    if zoom_out:
        driver.execute_script('document.body.style.zoom = "50%"')
    driver.save_screenshot(save_path)
    driver.quit()


# ------- OpenReview Helper Functions -------------------------------------------

def get_author_lst_orforum(url):
    """
    Given an openreview URL that points to paper page, 
    return list of author names
    """
    resp = requests.get(url)
    assert resp.encoding == 'utf-8'
    start_str = '"authors":{"value":['
    for item in resp.iter_lines():
        item_str = item.decode(encoding=resp.encoding)
        if start_str in item_str:
            item_str = item_str[item_str.index(start_str) + len(start_str):]
            item_str = item_str[0:item_str.index(']},"authorids":{"value":[')]
            name_list = item_str.split(",")
            name_list = [author.replace('"', "") for author in name_list]
            return name_list
    raise ValueError("get_author_ids_orforum cannot find authorids from openreview page")


def get_author_ids_orforum(url):
    """
    Given an openreview URL that points to paper page, return list of author ids
    """
    resp = requests.get(url)
    assert resp.encoding == 'utf-8'
    start_str = 'authorids":{"value":['
    for item in resp.iter_lines():
        item_str = item.decode(encoding=resp.encoding)
        if start_str in item_str:
            item_str = item_str[item_str.index(start_str) + len(start_str):]
            item_str = item_str[0:item_str.index("]}")]
            id_list = item_str.split(",")
            id_list = [authorid.replace('"', "") for authorid in id_list]
            return id_list
    raise ValueError("get_author_ids_orforum cannot find authorids from openreview page")


def _process_openreview_name_lst(name_lst):
    """Procress name list returned from openreview api"""
    # ignore names that don't have a username
    name_lst = [name for name in name_lst if "username" in name]
    assert len(name_lst) >= 1
    if len(name_lst) == 1:
        names = [{"fullname": name_lst[0]["fullname"],
                  "preferred": True,
                  "username": name_lst[0]["username"]}]
    else:
        names = [{"fullname": name["fullname"],
                  "preferred": name["preferred"] if "preferred" in name else None,
                  "username": name["username"]} for name in name_lst]
    return names


def return_author_name_or(client, authorid):
    """
    Given a authorid from OpenReview, output name from profile
    """
    profile = client.get_profile(authorid)
    return [AuthorNameOR(item) for item in _process_openreview_name_lst(profile.content["names"])]


def return_author_info_or(client, authorid):
    """
    Given a authorid from OpenReview, output information from their profile
    (names, career & education history, and personal links)
    """
    profile = client.get_profile(authorid)
    if not profile.active:
        raise ValueError("return_author_info_or: found inactive profile")

    # --------------- career and education history ---------------------------
    career_edu = []
    for history in profile.content["history"]:
        # ['position', 'start', 'end', 'institution']
        career_edu.append({"position": history["position"] if "position" in history else None, # string
                           "start": history["start"] if "start" in history else None, # year
                           "end": history["end"] if "end" in history else None, # year, but could be none
                           "institution": history["institution"]["name"]})
    info = {"career_edu_history": career_edu}
    
    # ----- Names ----------------------------
    info["names"] = _process_openreview_name_lst(profile.content["names"])

    # ---- Personal Links --------------
    info["personal_links"] = {"homepage": profile.content["homepage"] if "homepage" in profile.content.keys() else None,
                              "gscholar": profile.content["gscholar"] if "gscholar" in profile.content.keys() else None,
                              "dblp": profile.content["dblp"] if "dblp" in profile.content.keys() else None,
                              "orcid": profile.content["orcid"] if "orcid" in profile.content.keys() else None,
                              "linkedin": profile.content["linkedin"] if "linkedin" in profile.content.keys() else None,
                              "semanticScholar": profile.content["semanticScholar"] if "semanticScholar" in profile.content.keys() else None,
                              "aclanthology": profile.content["aclanthology"] if "aclanthology" in profile.content.keys() else None}
    return info