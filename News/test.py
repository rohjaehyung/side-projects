from newsapi import NewsApiClient

# Init
newsapi = NewsApiClient(api_key='9c982add42654ac38b673a8c3f714c9d')

# /v2/top-headlines
top_headlines = newsapi.get_top_headlines(q='bitcoin',
                                          sources='bbc-news,the-verge',
                                          category='business',
                                          language='en',
                                          country='us')
