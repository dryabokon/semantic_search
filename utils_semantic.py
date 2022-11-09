#https://towardsdatascience.com/quick-fire-guide-to-multi-modal-ml-with-openais-clip-2dad7e398ac0
import cv2
import os
import numpy
import pandas as pd
from os import listdir
import fnmatch
import requests
from PIL import Image
# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.insert(1, './tools/')
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_image
import tools_tensor_view
# ----------------------------------------------------------------------------------------------------------------------
class Semantic_proc:
    def __init__(self,folder_out,cold_start=True):
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.device = 'cpu' #"cuda" if torch.cuda.is_available() else  ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model_id = "openai/clip-vit-base-patch32"
        self.folder_out=folder_out
        self.hex_mode = True
        self.token_size = 512
        if cold_start:
            self.cold_start()

        return
# ----------------------------------------------------------------------------------------------------------------------
    def cold_start(self):
        from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
        self.tokenizer = CLIPTokenizerFast.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_filenames(self,path_input, list_of_masks):
        local_filenames = []
        for mask in list_of_masks.split(','):
            res = listdir(path_input)
            if mask != '*.*':
                res = fnmatch.filter(res, mask)
            local_filenames += res

        return numpy.sort(numpy.array(local_filenames))
# ----------------------------------------------------------------------------------------------------------------------
    def tokenize_images(self, folder_images,batch_size=100):

        filenames_images = self.get_filenames(folder_images, '*.png,*.jpg')
        batch_iter = 0
        while batch_iter < len(filenames_images):
            batch_filenames = filenames_images[batch_iter:batch_iter + batch_size]
            images = [cv2.imread(folder_images + filename) for filename in batch_filenames]
            inputs = self.processor(text=None,images=images,return_tensors='pt',padding=True)['pixel_values']
            img_emb = self.model.get_image_features(inputs).to(self.device).detach().numpy()
            df = pd.DataFrame(img_emb)
            if self.hex_mode:
                df = tools_DF.to_hex(df)

            df = pd.concat([pd.DataFrame({'image': batch_filenames}), df], axis=1)

            if batch_iter == 0:
                mode, header = 'w+',True
            else:
                mode, header = 'a', False

            df.to_csv(self.folder_out + 'tokens_%s.csv' % folder_images.split('/')[-2], index=False, float_format='%.4f', mode=mode,header=header)
            batch_iter += batch_size

        return
# ----------------------------------------------------------------------------------------------------------------------
    def tokenize_URLs_images(self,URLs,captions=None,do_save=True):

        if captions is None:
            captions = ['']*len(URLs)

        filename_out = self.folder_out + 'tokens.csv'
        if os.path.isfile(filename_out):
            os.remove(filename_out)

        for i,(URL,caption) in enumerate(zip(URLs,captions)):
            filename_image = '%06d.jpg' % i
            try:
                response = requests.get(URL,stream=True,timeout=2,allow_redirects=False)
            except:
                print(i, 'Timeout ', URL)
                continue

            if not response.ok:
                print(i, 'Bad response ', URL)
                continue

            try:
                image = cv2.cvtColor(numpy.array(Image.open(response.raw)), cv2.COLOR_RGB2BGR)
            except:
                print(i, 'Bad payload ', URL)
                continue
            if do_save:
                cv2.imwrite(self.folder_out+filename_image,image)

            print(i,'OK ',URL)
            inputs = self.processor(text=None, images=image, return_tensors='pt', padding=True)['pixel_values']
            img_emb = self.model.get_image_features(inputs).to(self.device).detach().numpy()
            df = pd.DataFrame(img_emb)
            if self.hex_mode:
                df = tools_DF.to_hex(df)

            df = pd.concat([pd.DataFrame({'image': [filename_image]}), df,pd.DataFrame({'caption': [caption]})], axis=1)

            if os.path.isfile(filename_out):
                mode, header = 'a', False
            else:
                mode, header = 'w+', True

            df.to_csv(filename_out, index=False, float_format='%.4f', mode=mode, header=header)


        return
# ----------------------------------------------------------------------------------------------------------------------
    def tokenize_words(self, filename_words,batch_size=50):
        words = [w for w in pd.read_csv(filename_words,header=None,sep='\t').values[:, 0]]

        batch_iter = 0
        while batch_iter<len(words):
            batch_words = words[batch_iter:batch_iter+batch_size]
            inputs = self.tokenizer(batch_words, return_tensors="pt",padding=True)
            text_emb = self.model.get_text_features(**inputs).to(self.device).detach().numpy()
            df = pd.DataFrame(text_emb)
            if self.hex_mode:
                df = tools_DF.to_hex(df)

            df = pd.concat([pd.DataFrame({'words': batch_words}), df], axis=1)
            mode ='w+' if batch_iter==0 else 'a'
            header = True if batch_iter==0 else False
            df.to_csv(self.folder_out + 'tokens_%s.csv'%filename_words.split('/')[-1].split('.')[0], index=False, float_format='%.4f',mode=mode,header=header)
            batch_iter+=batch_size

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_top_items(self, items, confidences, top_n=3):
        t_df = pd.DataFrame({'items':items,'conf':confidences})
        t_df = t_df.sort_values(by=t_df.columns[1], ascending=False)[:top_n]
        items = [i for i in t_df.iloc[:,0]]
        confidences = [i for i in t_df.iloc[:,1]]
        return items,confidences
# ----------------------------------------------------------------------------------------------------------------------
    def similarity_to_description(self,df,top_n=3):

        with open(self.folder_out + "descript.ion", mode='w+') as f_handle:
            for r in range(df.shape[0]):
                items,confidences = self.get_top_items(df.columns[1:],df.iloc[r,1:],top_n=top_n)
                str_items = ' '.join([item + '(%d)' % (100 * confidence) for item,confidence in zip(items,confidences)])

                f_handle.write("%s %s\n" % (df.iloc[r,0], str_items))
            f_handle.close()

        return
# ----------------------------------------------------------------------------------------------------------------------
    def preprocess_tokens(self, df_tokens):
        if isinstance(df_tokens, str):
            df = pd.read_csv(df_tokens)
        else:
            df = df_tokens

        df_temp = df.iloc[:, 1:1 + self.token_size]
        if self.hex_mode:
            df_temp = tools_DF.from_hex(df_temp)

        df_temp = df_temp / numpy.linalg.norm(df_temp, axis=0)
        df = pd.concat([df.iloc[:,0],df_temp],axis=1)

        return df
# ----------------------------------------------------------------------------------------------------------------------
    def tokens_similarity(self, filename_tokens1, filename_tokens2, top_n=3):

        df1 = self.preprocess_tokens(filename_tokens1)
        df2 = self.preprocess_tokens(filename_tokens2)
        names1 = df1.iloc[:,0].copy()
        df_similarity = df2.iloc[:,0].copy()

        df1 = df1.iloc[:, 1:1 + self.token_size]
        df2 = df2.iloc[:, 1:1 + self.token_size]

        for index, row in df1.iterrows():
            df_similarity = pd.concat([df_similarity,df2.dot(row).rename(names1[index])],axis=1)

        #df_similarity.to_csv(self.folder_out + 'similarity.csv', index=False, float_format='%.2f')

        self.similarity_to_description(df_similarity,top_n=top_n)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def search_images(self, query_text, filename_tokens_images, filename_tokens_words=None, top_n=5):

        df_images = self.preprocess_tokens(filename_tokens_images)
        df_words = pd.read_csv(filename_tokens_words)
        df_words = tools_DF.apply_filter(df_words,df_words.columns[0],query_text)
        df_words = self.preprocess_tokens(df_words)

        if  df_words.shape[0]==1:
            row = pd.Series(df_words.iloc[0, 1:1 + self.token_size],name=query_text)
            mat = df_images.iloc[:, 1:1 + self.token_size]

            df_similarity = pd.DataFrame({'P':mat.dot(row).values},index=df_images.iloc[:,0]).T
            df_similarity['text']=query_text
            df_similarity = pd.concat([df_similarity.iloc[:,-1],df_similarity.iloc[:,:-1]],axis=1)

        else:
            return

        #df_similarity.to_csv(self.folder_out + 'similarity.csv', index=False, float_format='%.2f')
        filenames_images,confidences = self.get_top_items(df_similarity.columns[1:],df_similarity.iloc[0,1:],top_n=top_n)


        return filenames_images
# ----------------------------------------------------------------------------------------------------------------------
    def compose_thumbnails(self,folder_in,filenames_images,fiename_out):

        small_width,small_height = 320,240
        tensor = [tools_image.smart_resize(cv2.imread(folder_in+filename),small_height,small_width) for filename in filenames_images]
        image = tools_tensor_view.tensor_color_4D_to_image(numpy.transpose(numpy.array(tensor), (1, 2, 3, 0)))
        cv2.imwrite(self.folder_out+fiename_out,image)

        return
# ----------------------------------------------------------------------------------------------------------------------

