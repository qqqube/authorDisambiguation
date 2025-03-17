import argparse
import os
import shutil
from utils import *
from openai import OpenAI
import pandas as pd
from ast import literal_eval
import nltk
import openreview
from datetime import date


def _do_part_one(client, pdf_dir, csv_save_path, tmp_dir):
    """
    Extract title, author list, and affiliation list from each paper
    """
    # ---------------- load prompts -------------------------
    with open("prompts/title_extract.txt", "r") as file:
        title_prompt = file.read()
    with open("prompts/author_names.txt", "r") as file:
        author_names_prompt = file.read()
    with open("prompts/institution_names.txt", "r") as file:
        institution_names_prompt = file.read()
    # -------------------------------------------------------

    out = []
    for paper_pdf in os.listdir(pdf_dir):
        if not paper_pdf.endswith(".pdf"):
            continue

        curr_id = paper_pdf.replace(".pdf", "")
        curr_path = os.path.join(pdf_dir, paper_pdf)

        print(f"curr_id: {curr_id}")

        # --- take a screenshot of the first page of the paper -----
        img_path = os.path.join(tmp_dir, f"{curr_id}.jpg")
        save_first_pdf_page_as_img(curr_path, img_path)
        base64_image = encode_image(img_path)

        # ---- gpt4o inference calls  ----------------
        title = gpt4o(client, title_prompt, base64_image).strip()
        author_names = gpt4o(client, author_names_prompt, base64_image).split("|")
        author_names = [name.strip() for name in author_names]
        institution_names = gpt4o(client, institution_names_prompt, base64_image).split("|")
        institution_names = [name.strip() for name in institution_names]

        out.append({"id": curr_id, "title": title,
                    "author_names": author_names,
                    "institution_names": institution_names})
    
    df = pd.DataFrame.from_records(out)
    df.to_csv(csv_save_path, index=False)


def _openreview_paper_search(serper, title):
    """ 
    Given the title of a paper, determine if the paper is
    hosted on OpenReview or not
    Will return either URL of openreview paper page or None
    """
    _query = f"openreview {title}"
    print(f"_openreview_paper_search {_query}")
    results = serper.query(_query)
    for result in results:
        if not result.link.startswith("https://openreview.net/forum?id="):
            continue
        if "=" in result.link.replace("https://openreview.net/forum?id=", ""):
            # ignore links that have other parameters passed in (in addition to id)
            continue
        str_distance = nltk.edit_distance(result.title.lower(), title.lower())
        if str_distance <= 1:
            return result.link
        if result.title.endswith("..."):
            result_title = result.title.replace("...", "").strip()
            # we expect result_title to be a substring of title, but title was extracted by an LLM
            # so it might contain a few mistakes
            str_distance = nltk.edit_distance(result_title.lower(), title[0:len(result_title)].lower())
            if str_distance > 1:
                continue
            return result.link
        if result.title.endswith("- OpenReview"):
            result_title = result.title.replace("- OpenReview", "").strip()
            # we expect an exact match, but title was given by LLM so it might contain a few mistaks
            str_distance = nltk.edit_distance(result_title.lower(), title.lower())
            if str_distance > 1:
                continue
            return result.link
        if ":" in title:
            _, last = title.split(":", 1)
            str_distance = nltk.edit_distance(last.lower(), result.title.lower())
            if str_distance > 1:
                continue
            return result.link
    return None


def _find_paper_on_website(serper, _url, paper_title):
    """Find paper on website"""
    _query = f"{_url} {paper_title}"
    new_results = serper.query(_query)
    for new_result in new_results:
        if _url not in new_result.link:
            continue
        if paper_title.lower() not in new_result.snippet.lower():
            continue
        return 1
    return 0


def _passes(personal_website):
    """Return true if personal website is not in list of bad links"""
    EXCLUDE = ["scholar.google", "openreview.net", "linkedin.com",
               "underline.io", "aclanthology.org", "researchgate.net",
               "semanticscholar.org", "arxiv.org", "dblp.org",
               "paperswithcode.com", "facebook.com", "github.com",
               "x.com", "wandb.ai", "huggingface.co", "ieee.org",
               "twitter.com", "acm.org", "rocketreach.co"]
    for item in EXCLUDE:
        if item in personal_website:
            return False
    return True


def _find_personal_website(serper, openai_client, author_name,
                           institution_name, paper_title, tmp_dir):
    """
    Attempts to find the personal website of the author
    """
    _query = f"{author_name} {institution_name}"
    print(f"_find_personal_website serper: {_query}")
    results = serper.query(_query)
    results = [result for result in results if _passes(result.link)]
    URLS = []
    for result in results:
        if author_name.lower() not in result.title.lower():
            continue
        if not (result.link.endswith("github.io") or ".edu" in result.link):
            continue
        URLS.append(result.link)
    if len(URLS) > 0:
        for _url in URLS:
            rtc = _find_paper_on_website(serper, _url, paper_title)
            if rtc == 1:
                return _url
    # either URLS is empty or no link was returned when URLS had at least one item
    with open("prompts/webpage_screenshot.txt", "r") as file:
        prompt = file.read()
    for result in results:
        if result.link in URLS:
            continue
        if author_name.lower() in result.title.lower():
            # take a screenshot of the URL
            image_path = os.path.join(tmp_dir, "webpage_screenshot.png")
            capture_webpage(result.link, image_path)

            # Getting the Base64 string
            base64_image = encode_image(image_path)

            # ---- create prompt -----------------------------------
            curr_prompt = prompt.replace("{{author name}}", author_name)
            curr_prompt = curr_prompt.replace("{{website url}}", result.link)
            print(curr_prompt)

            # ---- gpt-4o inference call -----
            answer = gpt4o(openai_client, curr_prompt, base64_image).strip().lower()
            if answer == "yes":
                rtc = _find_paper_on_website(serper, result.link, paper_title)
                if rtc == 1:
                    return result.link
    return None


def _find_gscholar_url(serper, author_name, paper_title):
    """Gind the google scholar profile"""
    _query = f"{author_name} google scholar {paper_title}"
    print(f"_find_gscholar_url serper: {_query}")
    results = serper.query(_query)
    for result in results:
        if author_name not in result.title:
            continue
        if "scholar.google" not in result.link:
            continue
        if paper_title.lower() not in result.snippet.lower():
            continue
        return result.link
    return None


def _get_personal_website_affiliation(openai_client, personal_website, author_name, tmp_dir):
    """
    Get affiliation from personal website
    """
    # take a screenshot of the URL
    image_path = os.path.join(tmp_dir, "webpage_screenshot.png")
    capture_webpage(personal_website, image_path)

    # load the prompt
    with open("prompts/get_affiliation_from_personal_website.txt", "r") as file:
        prompt = file.read()
    
    # Getting the Base64 string
    base64_image = encode_image(image_path)

    todays_date = date.today()
    month = todays_date.strftime("%B")
    year = todays_date.strftime("%Y")

    # ---- create prompt -----------------------------------
    curr_prompt = prompt.replace("{{author name}}", author_name)
    curr_prompt = curr_prompt.replace("{{date_month}}", f"{month} {year}")
    print(curr_prompt)
    # --------------------------------------------------------

    # ---- gpt-4o inference call -----
    answer = gpt4o(openai_client, curr_prompt, base64_image).strip()
    if answer.lower() != "no answer":
        return answer
    return None


def _get_gscholar_affiliation(openai_client, gscholar_url, author_name, tmp_dir):
    """Get affiliation from google scholar"""
    
    # take a screenshot of the URL
    image_path = os.path.join(tmp_dir, "webpage_screenshot.png")
    capture_webpage(gscholar_url, image_path)

    # load the prompt
    with open("prompts/get_affiliation_from_gscholar.txt", "r") as file:
        prompt = file.read()

    # Getting the Base64 string
    base64_image = encode_image(image_path)

    # ---- create prompt -----------------------------------
    curr_prompt = prompt.replace("{{author name}}", author_name)
    print(curr_prompt)
    # --------------------------------------------------------

    # ---- gpt-4o inference call -----
    answer = gpt4o(openai_client, curr_prompt, base64_image).strip()
    if answer.lower() != "no answer":
        return answer
    return None


def _do_part_two(client, or_client, serper, extracted_attr_path, tmp_dir, save_dir):

    df = pd.read_csv(extracted_attr_path)
    df["author_names"] = df["author_names"].apply(literal_eval)
    df["institution_names"] = df["institution_names"].apply(literal_eval)

    openreview_id_to_author = {}
    website_to_author = {}
    gscholar_to_author = {}
    papers = {} # map paper ID to paper instance
    extra_author_instances = []
    for _, row in df.iterrows():

        # determine if paper is hosted on OpenReview
        openreview_url = _openreview_paper_search(serper, row["title"])

        # create paper instance for each row (add to papers dictionary)
        paper_inst = Paper({**{index_item : row[index_item] for index_item in row.index},
                            "openreview_url": openreview_url})
        papers[paper_inst.id] = paper_inst

        if paper_inst.hosted_on_openreview():
            try:
                authorid_list = get_author_ids_orforum(openreview_url)
                author_list = get_author_lst_orforum(openreview_url)
            except Exception as e:
                print(f"EXCEPTION get_author_ids_orforum: {e}")
                paper_inst.openreview_url = None
                continue

            for author_name, authorid in zip(author_list, authorid_list):
                ###
                # Create an Author instance for each author id
                #   Check to see if we've already created an author instance for the id of the PREFERRED name
                #      before creating a new one each time.
                #   Note that exceptions may occur when getting info from profile (e.g., if profile is inactive, 
                #      but in this case we can still get the user's list of names 
                #      which is done in the exception handling code). Just create 
                #      a new author instance.
                
                if not authorid.startswith('https://dblp.org'):
                    try:
                        or_author_info = return_author_info_or(or_client, authorid)
                        author_or_info_inst = AuthorOpenReviewProfile(or_author_info)
                        preferred_authorid = author_or_info_inst.preferred_author_id
                        if preferred_authorid in openreview_id_to_author:
                            author_inst = openreview_id_to_author[preferred_authorid]
                        else:
                            author_inst = Author(openreview_profile=author_or_info_inst)
                            openreview_id_to_author[preferred_authorid] = author_inst
                        
                        homepage = author_or_info_inst.get_homepage()
                        gscholar_url = author_or_info_inst.get_gscholar()
                        if homepage:
                            website_to_author[homepage] = author_inst
                        if gscholar_url:
                            gscholar_to_author[gscholar_url] = author_inst
                    except Exception:
                        print(f"LOOP 1 EXCEPTION: {authorid}")
                        try:
                            name_lst = return_author_name_or(or_client, authorid)
                            preferred_authorid = get_preferred_name(name_lst).username
                            if preferred_authorid in openreview_id_to_author:
                                author_inst = openreview_id_to_author[preferred_authorid]
                            else:
                                author_inst = Author(name_lst=name_lst)
                                openreview_id_to_author[preferred_authorid] = author_inst
                        except Exception:
                            websearch_urls = WebSearchURLs(author_name)
                            author_inst = Author(websearch_urls=websearch_urls)
                            extra_author_instances.append(author_inst)
                else:
                    websearch_urls = WebSearchURLs(author_name)
                    author_inst = Author(websearch_urls=websearch_urls)
                    extra_author_instances.append(author_inst)

                # add paper to author instance
                author_inst.add_paper(paper_inst)
                # add author instance to paper
                paper_inst.add_author(author_inst)

    authors_without_any_urls = []
    for author_inst in list(openreview_id_to_author.values()) + extra_author_instances:
        # if we don't have author_inst's OpenReview profile information, 
        # then find the personal website or google scholar URL by
        # matching with just one of their papers from OpenReview
        if author_inst.has_openreview_profile():
            continue
        found_personal_website = False
        found_gscholar = False
        websearch_urls = WebSearchURLs(author_inst.fullname)
        for paper in author_inst.get_papers():
            if not found_gscholar:
                gscholar_url = _find_gscholar_url(serper, author_inst.fullname,
                                                  paper.title)
                if gscholar_url is not None:
                    found_gscholar = True
                    gscholar_to_author[gscholar_url] = author_inst
                    gscholar_affiliation = _get_gscholar_affiliation(client, gscholar_url,
                                                                     author_inst.fullname,
                                                                     tmp_dir)
                    websearch_urls.set_gscholar_url(gscholar_url, gscholar_affiliation)
            for institution_name in paper.institution_names:
                if not found_personal_website:
                    personal_website = _find_personal_website(serper, client,
                                                              author_inst.fullname,
                                                              institution_name,
                                                              paper.title, tmp_dir)
                    if personal_website is not None:
                        found_personal_website = True
                        website_to_author[personal_website] = author_inst
                        personal_website_affiliation = _get_personal_website_affiliation(client,
                                                                                         personal_website,
                                                                                         author_inst.fullname,
                                                                                         tmp_dir)
                        websearch_urls.set_homepage_url(personal_website, personal_website_affiliation)
                if found_personal_website:
                    break
            if found_gscholar and found_personal_website:
                break
        if found_gscholar or found_personal_website:
            author_inst.add_websearch_urls(websearch_urls)
        else:
            authors_without_any_urls.append(author_inst)

    print(f"len(authors_without_any_urls) = {len(authors_without_any_urls)}")
    for _, paper in papers.items():
        # if paper is not hosted on OpenReview, do Step 3B (find either personal website or google scholar profile)
        # Could create new authors here, but first check website_to_author and gscholar_to_author
        #     to find existing authors
        if paper.hosted_on_openreview():
            continue
        for author_name in paper.author_names:
            websearch_urls = WebSearchURLs(author_name)
            gscholar_url = _find_gscholar_url(serper, author_name, paper.title)
            if (gscholar_url is not None) and (gscholar_url in gscholar_to_author):
                author_inst = gscholar_to_author[gscholar_url]
                if author_inst.homepage_url() is not None:
                    author_inst.add_paper(paper)
                    paper.add_author(author_inst)
                    continue
            # At this point...
            # (1) Found google scholar URL and author instance exists but no homepage URL yet 
            # (2) Found google scholar URL but author instance doesn't exist 
            # (3) Didn't find a google scholar URL
            
            # ---- Try to find personal website -------
            personal_website = None
            for institution_name in paper.institution_names:
                personal_website = _find_personal_website(serper, client, author_name,
                                                          institution_name, paper.title, tmp_dir)
                if personal_website is not None:
                    break
            
            # Case (1) 
            if (gscholar_url is not None) and (gscholar_url in gscholar_to_author):
                author_inst = gscholar_to_author[gscholar_url]
                author_inst.add_paper(paper)
                paper.add_author(author_inst)
                if personal_website is not None:
                    personal_website_affiliation = _get_personal_website_affiliation(client, personal_website, author_inst.fullname, tmp_dir)
                    websearch_urls.set_homepage_url(personal_website, personal_website_affiliation)
                    author_inst.add_websearch_urls(websearch_urls)
                    website_to_author[personal_website] = author_inst
                continue
            # Case (2)
            elif gscholar_url is not None:
                gscholar_affiliation = _get_gscholar_affiliation(client, gscholar_url, author_name, tmp_dir)
                websearch_urls.set_gscholar_url(gscholar_url, gscholar_affiliation)
            # Case (2) and (3)  
            author_inst = None
            if personal_website is not None and personal_website in website_to_author:
                author_inst = website_to_author[personal_website]
                author_inst.add_websearch_urls(websearch_urls)
            else:
                if personal_website is not None:
                    personal_website_affiliation = _get_personal_website_affiliation(client, personal_website,
                                                                                     author_name, tmp_dir)
                    websearch_urls.set_homepage_url(personal_website, personal_website_affiliation)
                author_inst = Author(websearch_urls=websearch_urls)
            author_inst.add_paper(paper)
            paper.add_author(author_inst)

            # update mappings
            if author_inst.homepage_url() is not None:
                website_to_author[author_inst.homepage_url()] = author_inst
            if author_inst.gscholar_url() is not None:
                gscholar_to_author[author_inst.gscholar_url()] = author_inst

            # authors without urls
            if not author_inst.has_urls():
                authors_without_any_urls.append(author_inst)
    
    print(f"len(authors_without_any_urls) = {len(authors_without_any_urls)}")
    all_authors = list(openreview_id_to_author.values()) + list(website_to_author.values()) + list(gscholar_to_author.values()) + extra_author_instances
    all_authors = list(set(all_authors))
    all_authors_with_urls = [author_inst for author_inst in all_authors if author_inst.has_urls()]
    num_authors_assigned = 0
    for author_inst in authors_without_any_urls:
        candidates = []
        for author_ref in all_authors_with_urls:
            if author_inst.fullname != author_ref.fullname:
                continue
            # compare coauthors
            author_inst_coauthors = set(author_inst.get_coauthors())
            author_ref_coauthors = set(author_ref.get_coauthors())
            coauthor_intersection = author_inst_coauthors.intersection(author_ref_coauthors)
            if len(coauthor_intersection) > 0:
                candidates.append((len(coauthor_intersection), author_ref))
        if len(candidates) > 0:
            max_intersection_size = max([item[0] for item in candidates])
            candidates = [item for item in candidates if item[0] == max_intersection_size]
            candidates[0][1].merge_author(author_inst, max_intersection_size)
            num_authors_assigned += 1
    print(f"assigned {num_authors_assigned} out of {len(authors_without_any_urls)}")

    # ------- save all author information ---------------
    all_authors = [author_inst for author_inst in authors_without_any_urls if not author_inst.has_urls()] + all_authors_with_urls
    all_authors = list(set(all_authors))
    author_records = []
    assignment_records = []
    for author_id, author_inst in enumerate(all_authors):
        author_record = {"author_id": author_id,
                         "author_fullname": author_inst.fullname,
                         "openreview_url": author_inst.openreview_url(),
                         "gscholar_url": author_inst.gscholar_url(),
                         "homepage_url": author_inst.homepage_url(),
                         "affiliation": author_inst.affiliation()}
        author_records.append(author_record)

        for pdf_name, confidence_score in author_inst.get_merged_papers():
            assignment_record = {"author_id": author_id,
                                 "pdf_name": pdf_name,
                                 "confidence_score": confidence_score}
            assignment_records.append(assignment_record)
        for paper_inst in author_inst.get_papers():
            assignment_record = {"author_id": author_id,
                                 "pdf_name": paper_inst.pdf_name(),
                                 "confidence_score": 1.0}
            assignment_records.append(assignment_record)
    author_df = pd.DataFrame.from_records(author_records)
    assignment_df = pd.DataFrame.from_records(assignment_records)
    print(f"author_df.shape = {author_df.shape}")
    print(f"assignment_df.shape = {assignment_df.shape}")
    author_df.to_csv(os.path.join(save_dir, "authors.csv"), index=False)
    assignment_df.to_csv(os.path.join(save_dir, "assignments.csv"), index=False)


if __name__ == "__main__":
    
    # load arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, choices=[1, 2])
    parser.add_argument("--pdf_dir", type=str) # path to directory containing PDFs where the name of each PDF is <id>.pdf
    parser.add_argument("--tmp_dir", type=str, default="tmp") # directory to store files in (will be deleted at the end)
    parser.add_argument("--save_dir", type=str, default="out") # directory to save output (only used in part 2)
    parser.add_argument("--credentials_path", type=str, default="../credentials.ini") # path to credentials file
    args = parser.parse_args()

    # ---- create OpenAI client ------------
    USERNAME, PASSWORD, SERPER_API_KEY, OPENAI_API_KEY = get_credentials("../credentials.ini")
    client = OpenAI(api_key=OPENAI_API_KEY)

    # ---- create tmp_dir ------------------
    if os.path.exists(args.tmp_dir):
        shutil.rmtree(args.tmp_dir)
        print(f"Deleted '{args.tmp_dir}'")
    os.mkdir(args.tmp_dir)
    print(f"Successfully created {args.tmp_dir}")

    # ----- create save_dir ---------------
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
        print(f"Deleted {args.save_dir}")
    os.mkdir(args.save_dir)
    print(f"Successfully creted {args.save_dir}")
    
    if args.part == 1:
        # --- extract title, author list, and affiliation list from each paper --------
        _do_part_one(client, args.pdf_dir, f"{args.pdf_dir}.csv", args.tmp_dir)
    elif args.part == 2:
        serper = Serper(SERPER_API_KEY)
        or_client = openreview.api.OpenReviewClient(
                        baseurl='https://api2.openreview.net',
                        username=USERNAME, password=PASSWORD
                    )
        _do_part_two(client, or_client, serper, f"{args.pdf_dir}.csv", args.tmp_dir, args.save_dir)

    # ----- delete tmp_dir ------------------
    #shutil.rmtree(args.tmp_dir)
    #print(f"Deleted '{args.tmp_dir}'")