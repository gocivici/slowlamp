## Cover Library
```cover``` is a library that can be used to save and retrieve the colors captured by the slowlamp. Each data capture (```Record```) is saved in a 4x3  grid.

### How to use

```Python
import cover

cover.save(
    color_vibrant, count_vc, 
    ambientColor1, count_ac_1, 
    ambientColor2, count_ac_2, 
    ambientColor3, count_ac_3, 
    ambientColor4 , count_ac_4, 
    current_timestamp
) #saves colors to archive.png

print(cover.retrieve()) #reads archive.png and returns data as a dictionary.
```
