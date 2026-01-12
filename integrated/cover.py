import os
import math
import time
import random
from PIL import Image


def encode_24bit_pixel(integer_value):
    #encode integer to 24 bit pixel in the form of (R,G,B,0)
    val = int(integer_value) & 0xFFFFFF # Mask to 24 bits
    R = val & 0xFF
    G = (val >> 8) & 0xFF
    B = (val >> 16) & 0xFF
    return (R, G, B, 0)

def decode_24bit_pixel(pixel):
    if len(pixel) < 3: return 0
    R, G, B = pixel[0], pixel[1], pixel[2]
    return R | (G << 8) | (B << 16)

def encode_color_pixel(color_array):
    #make color pixels opaque (make alpha channel 255)
    return tuple(map(int, color_array)) + (255,) 

def decode_32bit_timestamp(pixel):
    if len(pixel) < 4: return 0
    R, G, B, A = pixel
    return R | (G << 8) | (B << 16) | (A << 24)

def encode_32bit_timestamp(TimeStamp):
    #encode timestamp value to 32 bit pixel in the form of (R,G,B,A)
    val = int(TimeStamp)
    return (val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF, (val >> 24) & 0xFF)


# read existing image as grids of 4x3 and return existing blocks
def get_existing_blocks(filename, block_w=4, block_h=3):

    if not os.path.exists(filename):
        return []
    try:
        img = Image.open(filename).convert("RGBA")
    except:
        return []

    width, height = img.size
    cols_in_blocks = width // block_w
    rows_in_blocks = height // block_h
    
    blocks = []
    
    for r in range(rows_in_blocks):
        for c in range(cols_in_blocks):
            # Calculate the top-left corner of the current block
            left = c * block_w
            top = r * block_h
            right = left + block_w
            bottom = top + block_h
            
            # Crop 4x3 region
            crop = img.crop((left, top, right, bottom))
            pixels = list(crop.getdata())
            
            # check for empty padding
            if len(pixels) == 12 and pixels != [(0,0,0,0)]*12:
                 blocks.append(pixels)
                 
    return blocks


# main save function
def save(VibrantColor, pixelCountVC, color2, pixelCountColor2, 
                        color3, pixelCountColor3, color4, pixelCountColor4, 
                        color5, pixelCountColor5, TimeStamp):
    
    # define grid size
    BLOCK_W = 4
    BLOCK_H = 3
    BLOCKS_PER_ROW = 24  # new row after 24 blocks

    # create the New Data Block   
    new_block = [
        # Row 1
        encode_24bit_pixel(pixelCountVC), 
        encode_color_pixel(color2), 
        encode_24bit_pixel(pixelCountColor2), 
        encode_32bit_timestamp(TimeStamp),      

        # Row 2
        encode_color_pixel(color5), 
        encode_color_pixel(VibrantColor), 
        encode_color_pixel(color3), 
        encode_24bit_pixel(pixelCountColor5),   

        # Row 3
        encode_24bit_pixel(pixelCountColor4), 
        encode_color_pixel(color4), 
        encode_24bit_pixel(pixelCountColor3), 
        (75, 0, 130, 255)                        
    ]

    # get existing records
    all_blocks = get_existing_blocks("archive.png", BLOCK_W, BLOCK_H)
    
    # add new record
    all_blocks.append(new_block)
    
    # caşulate new canvas size
    total_blocks = len(all_blocks)
    grid_rows = math.ceil(total_blocks / BLOCKS_PER_ROW)
    grid_cols = min(total_blocks, BLOCKS_PER_ROW)
    
    img_w = grid_cols * BLOCK_W
    img_h = grid_rows * BLOCK_H
    
    new_img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    
    # add records to cavas

    for i, block_pixels in enumerate(all_blocks):
        row_idx = i // BLOCKS_PER_ROW
        col_idx = i % BLOCKS_PER_ROW
        
        x = col_idx * BLOCK_W
        y = row_idx * BLOCK_H
        
        temp_block = Image.new("RGBA", (BLOCK_W, BLOCK_H))
        temp_block.putdata(block_pixels)
        
        new_img.paste(temp_block, (x, y))
        
    new_img.save("archive.png")
    #print(f"Saved successfully. Total Blocks: {total_blocks}. Grid Size: {grid_cols}x{grid_rows} blocks ({img_w}x{img_h} pixels).")

#main retrieve function
def retrieve(filename="archive.png"):

    start_time = time.perf_counter()
    
    if not os.path.exists(filename):
        print(f"File '{filename}' not found.")
        return []

    blocks = get_existing_blocks(filename)
    results = []
    
    for i, block in enumerate(blocks):
        # Decode Data based on layout
        
        # Row 1
        count_vc    = decode_24bit_pixel(block[0])
        c_color2    = block[1][:3]
        count_c2    = decode_24bit_pixel(block[2])
        ts_val      = decode_32bit_timestamp(block[3])
        
        # Row 2
        c_color5    = block[4][:3]
        c_vibrant   = block[5][:3]
        c_color3    = block[6][:3]
        count_c5    = decode_24bit_pixel(block[7])
        
        # Row 3
        count_c4    = decode_24bit_pixel(block[8])
        c_color4    = block[9][:3]
        count_c3    = decode_24bit_pixel(block[10])
        
        # Build Dictionary for this block
        block_data = {
            "id": i + 1,
            "timestamp": ts_val,
            "vc": c_vibrant,
            "vc_px": count_vc,
            "ac1": c_color2,
            "ac1_px": count_c2,
            "ac2": c_color3,
            "ac2_px": count_c3,
            "ac3": c_color4,
            "ac3_px": count_c4,
            "ac4": c_color5,
            "ac4_px": count_c5
        }
        results.append(block_data)

    total_time = time.perf_counter() - start_time
    
    print(f"Retrieved {len(results)} records in {total_time:.4f} seconds.")
    
    return results

