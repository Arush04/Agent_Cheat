from bs4 import BeautifulSoup
import urllib.request

# Setting up configs
url = "https://codeforces.com/problemset/problem/2116/B"
user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7' 
headers={'User-Agent':user_agent,}

# Getting data
request=urllib.request.Request(url,None,headers)
response = urllib.request.urlopen(request)
soup = BeautifulSoup(response.read(), 'html.parser')

# Remove all css and js only keep html
for script in soup(["script", "style"]):
    script.extract()

# Finding Titles
x = soup.find_all("div", "title")
y = soup.find_all("div", "p")
print(y)
# print(x[0])
