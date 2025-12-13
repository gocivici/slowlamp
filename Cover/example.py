import cover
import time
import random

#get current timestamp
current_timestamp = int(time.time()) 

#generate andom colors
color_vibrant = [random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)] 
ambientColor1 = [random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)]    
ambientColor2 = [random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)]  
ambientColor3 = [random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)]   
ambientColor4 = [random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)] 

#generate random pixel counts
count_vc = random.randint(0, 1000000) 
count_ac_1 = random.randint(0, 1000000)  
count_ac_2 = random.randint(0, 1000000)  
count_ac_3 = random.randint(0, 1000000)  
count_ac_4 = random.randint(0, 1000000)  

cover.save(
    color_vibrant, count_vc, 
    ambientColor1, count_ac_1, 
    ambientColor2, count_ac_2, 
    ambientColor3, count_ac_3, 
    ambientColor4 , count_ac_4, 
    current_timestamp
) #saves colors to archive.png

print(cover.retrieve()) #reads archive.png and returns data as a dictionary.