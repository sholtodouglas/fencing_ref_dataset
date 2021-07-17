#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.makedirs('precut', exist_ok=True)
os.makedirs('staging', exist_ok=True)
os.makedirs('scoreboard_classifier_data', exist_ok=True)


# In[67]:


from pytube import YouTube
## Run this to download all the videos from youtube.

def dld(video):
    if f"{video.video_id}.mp4" not in os.listdir('precut'):
        print(f'dlding {video}')
        video.streams.filter(file_extension='mp4', progressive=True, res='360p', fps=25)[0].download(output_path = 'staging', filename= video.video_id, timeout=100)
        os.rename(f'staging/{video.video_id}.mp4',f'precut/{video.video_id}.mp4')
def download_vid(i):
    dld(YouTube(i))


from pytube import Playlist    
def download_playlist(p):
    p = Playlist(p)
    for video in p.videos:
        dld(video)


# In[61]:


os.listdir('precut')


# In[ ]:


text_file = open("playlists.txt", "r")
playlists = text_file.read().split('\n')
print("First 3 links:", playlists[:3])
text_file.close()

for p in playlists:
    download_playlist(p)


# In[ ]:




