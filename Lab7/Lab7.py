import cv2
import numpy as np
import matplotlib.pyplot as plt

def match_template_with_rotation_and_scaling(image, template, max_scale=2.0, scale_step=0.1, max_angle=360, angle_step=10, threshold=0.8):
    
    h_t, w_t = template.shape[:2]
    
    pyramid_levels = int(np.log2(max(image.shape[0], image.shape[1]) / min(w_t, h_t))) + 1
    pyramids = [image]
    
    for i in range(1, pyramid_levels):
        pyramids.append(cv2.pyrDown(pyramids[i - 1]))
    
    best_matches = []
    
    for pyramid in pyramids:
        h_img, w_img = pyramid.shape[:2]
        
        for scale in np.arange(1, max_scale, scale_step):
            resized_template = cv2.resize(template, (int(w_t * scale), int(h_t * scale)))
            h_t_resized, w_t_resized = resized_template.shape[:2]
            
            if h_t_resized > h_img or w_t_resized > w_img:
                continue
            
            for angle in np.arange(0, max_angle, angle_step):
                M = cv2.getRotationMatrix2D((w_t_resized / 2, h_t_resized / 2), angle, 1)
                rotated_template = cv2.warpAffine(resized_template, M, (w_t_resized, h_t_resized))
                
                result = cv2.matchTemplate(pyramid, rotated_template, cv2.TM_CCOEFF_NORMED)
            
                loc = np.where(result >= threshold)
                for pt in zip(*loc[::-1]):
                    best_matches.append((pt, scale, angle, result[pt[1], pt[0]]))
    
    result_image = image.copy()
    for match in best_matches:
        pt, scale, angle, score = match
        h_t_resized, w_t_resized = int(h_t * scale), int(w_t * scale)
        
        cv2.rectangle(result_image, pt, (pt[0] + w_t_resized, pt[1] + h_t_resized), (0, 255, 0), 2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Найдено {len(best_matches)} Совпадений')
    plt.show()

image = cv2.imread('stock_image.jpg')  
template = cv2.imread('part_of_image.jpg') 

match_template_with_rotation_and_scaling(image, template)
