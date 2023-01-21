import cv2
import os
import numpy as np
from xml.dom import minidom 



# DEPRECATED

images = ['earL', 'earR', 'body', 'face', 'crown', 'eyeL', 'eyeR', 'moustacheL', 'moustacheR', 'mouth']
dico = {
    'earL' : [], 
    'earR' : [], 
    'face' : [], 
    'eyeL' : [],
    'eyeR' : [],
    'moustacheL' : [],
    'moustacheR' : [],
    'mouth' : []
}
saved_points = []
img = None
part = 'Left ear base'
order = 'Left Center Right'

    
def click(event, x, y, flags, param):
        '''
        Handler for mouse click.
        Gets the coordinates clicked in the list saved_points.
        '''
        global saved_points, img, part, order
        
        if len(dico['earL']) < 3 :
            #left ear
            part = 'Left ear base'
            order = 'Left Center Right'
            temp = 'earL'
            
        elif len(dico['earR']) < 3 :
            #right ear
            part = 'Right ear base'
            order = 'Left Center Right'  
            temp = 'earR'
            
        elif len(dico['face']) < 4 :
            #face bounds
            part = 'Face bounds'
            order = 'Bottom Left Top Right'
            temp = 'face'
            
        elif len(dico['eyeL']) <  4 :
            #left iris
            part = 'Left iris bounds'
            order = 'Bottom Left Top Right'
            temp = 'eyeL'
            
        elif len(dico['eyeR']) <  4 :
            #right iris
            part = 'Right iris bounds'
            order = 'Bottom Left Top Right'
            temp = 'eyeR'
            
        elif len(dico['moustacheL']) < 2 :
            #left moustache
            part = 'Left moustache'
            order = 'Bottom Top'
            temp = 'moustacheL'
            
        elif len(dico['moustacheR']) < 2 :
            #right moustache
            part = 'Right moustache'
            order = 'Bottom Top'
            temp = 'moustacheR'
            
        elif len(dico['mouth']) < 4 :
            #mouth
            part = 'Mouth'
            order = 'Bottom Left Top Right'
            temp = 'mouth'

        if event == cv2.EVENT_LBUTTONDOWN:
            saved_points.append((x, y))
            dico[temp].append((x, y))
            

cv2.namedWindow('Avatar')
cv2.setMouseCallback("Avatar", click)
 

while True:
    img = cv2.imread('.\\avatar\\fuwa_glace.png', cv2.IMREAD_UNCHANGED)
    
    for i in range(len(saved_points)):
        cv2.circle(img, (int(saved_points[i][0]), int(saved_points[i][1])), 1, (0, 0, 255), 3)
        

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
    
    cv2.putText(img, f'{part} : {order}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Avatar", img)
    
    if len(dico['mouth']) == 4 :
        break
            
print(dico)

root = minidom.Document()
avatar = root.createElement('Avatar')
bottom = root.createElement('bottom')
left = root.createElement('left')
top = root.createElement('top')
center = root.createElement('center')    
right = root.createElement('right')

root.appendChild(avatar)
for i in range(len(dico)):
    
    keys = list(dico.keys())
    child = root.createElement(keys[i])
    bottom = root.createElement('bottom')
    left = root.createElement('left')
    top = root.createElement('top')
    center = root.createElement('center')
    right = root.createElement('right')
    
    
    avatar.appendChild(child)
    
    if keys[i] == 'earL' or keys[i] == 'earR' :
        left.setAttribute('x', str(dico[keys[i]][0][0]))
        left.setAttribute('y', str(dico[keys[i]][0][1]))
        center.setAttribute('x', str(dico[keys[i]][1][0]))
        center.setAttribute('y', str(dico[keys[i]][1][1]))
        right.setAttribute('x', str(dico[keys[i]][2][0]))
        right.setAttribute('y', str(dico[keys[i]][2][1]))
    elif keys[i] == 'face' or keys[i] == 'eyeL' or keys[i] == 'eyeR' or keys[i] == 'mouth': 
        bottom.setAttribute('x', str(dico[keys[i]][0][0])) 
        bottom.setAttribute('y', str(dico[keys[i]][0][1]))
        left.setAttribute('x', str(dico[keys[i]][1][0]))
        left.setAttribute('y', str(dico[keys[i]][1][1]))
        top.setAttribute('x', str(dico[keys[i]][2][0]))
        top.setAttribute('y', str(dico[keys[i]][2][1]))
        right.setAttribute('x', str(dico[keys[i]][3][0]))
        right.setAttribute('y', str(dico[keys[i]][3][1]))
    elif keys[i] == 'moustacheL' or keys[i] == 'moustacheR' :
        bottom.setAttribute('x', str(dico[keys[i]][0][0]))
        bottom.setAttribute('y', str(dico[keys[i]][0][1]))
        top.setAttribute('x', str(dico[keys[i]][1][0]))
        top.setAttribute('y', str(dico[keys[i]][1][1]))
       
    for element in [bottom, left, top, center, right] :
        child.appendChild(element)   
    
    

xml_str = root.toprettyxml('\t') 

with open('calibration_avatar.xml', 'w+') as f:
    f.write(xml_str)
    
    
    
    
    
# DEPRECATED    
#   
# Imports are needed but here is a function to parse this type of xml
# 
# def get_avatar_points(calibration_file = 'calibration_avatar.xml'):
#     '''
#     Function that parses the calibration file of the avatar to get the points of interest.
    
#     Args : 
#         - calibration_file (str) : the path of the xml calibration file 
    
#     Return : 
#         - dict (keys : Moving_part) : points of interest of the avatar
#     '''
#     # Initialize dictionary
#     dico = {}
    
#     # Parser le fichier avec minidom
#     doc = minidom.parse(calibration_file)
#     root = doc.getElementsByTagName(str(doc.firstChild.tagName))
#     avatar = root.item(0)
#     for moving_part in avatar.childNodes :
#         temp_list = []
#         temp_part = None
#         if moving_part.nodeType != Node.TEXT_NODE :
#             for point in moving_part.childNodes :
#                 if point.nodeType != Node.TEXT_NODE :
#                     if point.hasAttribute('x') :
#                         temp_list.append(np.array([int(point.attributes.item(0).value), int(point.attributes.item(1).value)]))
#                     else :
#                         temp_list.append(None)
#             temp_part = Moving_part(temp_list)            
#             new_key = {moving_part.tagName : temp_part}
#             dico.update(new_key)    
            
#     return dico
