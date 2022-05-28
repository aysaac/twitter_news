from seleniumwire import webdriver
from selenium.webdriver.common.by import By
wd=webdriver.Chrome('B:\\PycharmProjects\\twitter_news\\chromedriver\\chromedriver.exe')
#%%
mail='jecax71913@reimondo.com'
password='Gelatinaverde17.'
anti_feminazis='https://www.facebook.com/groups/936338860210148/'
g_feminazis='https://www.facebook.com/groups/134010054699212/'
#%%
wd.get('https://www.facebook.com/')
#%%
e_mail=wd.find_element(By.ID,"email")
passw=wd.find_element(By.ID,'pass')
enter=wd.find_element(By.ID,'u_0_d_kJ')
#%%
e_mail.send_keys(mail)
passw.send_keys(password)
enter.click()
#%%
wd.get(anti_feminazis)
#%%
list= wd.find_element(By.ID,'ssrb_top_nav_end')