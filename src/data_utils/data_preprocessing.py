
import py_vncorenlp
import pandas as pd
import multiprocessing
import os
py_vncorenlp.download_model(save_dir='/content')
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/content')

sign=[',', '.', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '<', '>', '"', "'", 
      '`', '...', '--', '-', '_', '*', '@', '#', '$', '%', '^', '&', '+', '=', '/', '\\', 
      '|', '~', '``', "''", '“', '”', '‘', '’', '«', '»', '„', '‟', '‹', '›', '〝', '〞', 
      '‒', '–', '—', '―',    '•', '·', '⋅', '°',':3','<3',':>',':v',':)','=)',':(','-.-','-_-']

def xoa_trung_lap(s):
    loop = ""
    i=1
    while i <= len(s)-1:
      try:
        if s[i]==s[i-1] and (i==len(s)-1 or s[i+1]==' '):
          j=i
          loop=s[i]
          while s[j-1] == s[j]:
            loop+=s[j]
            j-=1
          s = s.replace(loop, s[i])
        i+=1
      except:
        s='đéo biết nói gì nữa'
    return s
  
def scanerr(sentence,word_seg=True):
    # for t in sign:
    #     sentence=sentence.lower().replace(t,'')
    # sentence=xoa_trung_lap(sentence)
    if word_seg:
        return rdrsegmenter.word_segment(sentence)[0]
    else:
        return sentence

def main():
    data = pd.read_csv('/content/drive/MyDrive/DSAAA_2023/fucking/data1.csv')
    #data = data.rename(columns={'text1': 'id1_text', 'text2': 'id2_text', 'label': 'Label'})
   
    word_seg = True
    data['id1_text'] = data['id1_text'].apply(lambda x: scanerr(x, word_seg))
    data['id2_text'] = data['id2_text'].apply(lambda x: scanerr(x, word_seg))
    os.makedirs("data", exist_ok=True)
    data.to_csv('/content/data/data.csv')

if __name__ == '__main__':
    main()


# def main():
#     data = pd.read_csv('/content/drive/MyDrive/NLI/data/train.csv')
#     data = data.rename(columns={'text1': 'id1_text', 'text2': 'id2_text', 'label': 'Label'})

#     # Create a pool of worker processes
#     pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

#     # Apply the scanerr function to each row in the data
#     data['id1_text'] = pool.map(scanerr, data['id1_text'], data['word_seg'])
#     data['id2_text'] = pool.map(scanerr, data['id2_text'], data['word_seg'])

#     # Close the pool
#     pool.close()

#     data.to_csv('/content/data/data.csv')

# if __name__ == '__main__':
#     main()

    