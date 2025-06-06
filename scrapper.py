from bs4 import BeautifulSoup
import urllib.request

# Setting up configs
def parse_page(url):
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7' 
    headers={'User-Agent':user_agent,}

    title = ""
    body = ""
    input_list = []
    output_list = []
    # Getting data
    request=urllib.request.Request(url,None,headers)
    response = urllib.request.urlopen(request)
    soup = BeautifulSoup(response.read(), 'html.parser')

    # Remove all css and js only keep html
    for script in soup(["script", "style"]):
        script.extract()

    # Finding Title/ Question Name
    x = soup.find_all("div", "title")
    title = x[0].string
    # Finding Question / Output / Example [input, output] / Note
    content = soup.find("div", "ttypography")
    # For all discription text
    desc = content.find_all('p')

    for d in desc:
        body += d.string
        
    inputs = content.find("div", {"class": "input"})
    input_processed = inputs.find("pre")
    for i in input_processed:
        input_list.append(i.string)


    outputs = content.find("div", {"class": "output"})
    output_processed = outputs.find("pre")
    for o in output_processed:
        output_list.append(o.string)

    # Get Note data
    note = content.find("div", {"class": "note"})
    note_text = note.find("p").string
    note_lists = note.find_all("li")
    notes = []
    for n in note_lists:
        notes.append(n.string)
    print(title)
    print(body)
    print("INPUT")
    for i in input_list:
        print(i)
    print("OUTPUT")
    for o in output_list:
        print(o)
    print("NOTE")
    print(note_text)
    for n in notes:
        print(n)

url = "https://codeforces.com/problemset/problem/2116/B"
parser_page(url)
