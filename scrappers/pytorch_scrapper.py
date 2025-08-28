import os
import csv
from bs4 import BeautifulSoup
import requests
import time

def scrape_forum(url):
    """
    Scrape the forum for question-answer pair
    """ 
    final_url = f"{url}.json"
    header = {
        "Accept": "application/json"
    }
    request = requests.get(final_url, headers=header)
    data = request.json()
    post_data = data.get("post_stream").get("posts", [])[0]
    soup = BeautifulSoup(post_data.get("cooked"), 'html.parser')
    user_question = soup.get_text(strip=True)
    # answer = data.get("accepted_answer").get("excerpt")
    soup_a = BeautifulSoup(data.get("accepted_answer").get("excerpt"), 'html.parser')
    answer = soup_a.get_text(strip=True)
    question_user = post_data.get("username")
    answer_user = data.get("accepted_answer").get("username")

    return {
        "user_q": question_user,
        "question": user_question,
        "user_a": answer_user,
        "answer": answer
    }

def users_solved_ans(url):
    """
    Get list of solved urls 
    """
    user_solved_page = f"{url}.json"
    header = {
        "Accept": "application/json"
    }
    request = requests.get(url=user_solved_page, headers=header)
    data = request.json()
    all_posts = data.get("user_solved_posts")
    user_solved_posts = []
    for post in all_posts:
        post_link = post.get("url")
        user_solved_posts.append(post_link)
    return user_solved_posts

def save_to_csv(data, dataset_file):
    """
    Create/Append to dataset file
    """
    file_exists = os.path.isfile(dataset_file)
    with open(dataset_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user_q", "question", "user_a", "answer"])

        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


if __name__ == "__main__":
    dataset_file = "pytorch_qa_pair.csv"
    user_url = "https://discuss.pytorch.org/solution/by_user.json?username=ptrblck&offset=0&limit=20"
    post_urls = users_solved_ans(user_url)
    for post_url in post_urls:
        print(f"processing post {post_url}")
        data = scrape_forum(post_url)
        save_to_csv(data, dataset_file)
        time.sleep(1) # to handle request rate    

