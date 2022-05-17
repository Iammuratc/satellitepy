#### DETECTION DATA
class DetectionData(Data):
    
    def __init__(self,dataset_name):
        super(DetectionData, self).__init__(dataset_name)

    def save_patches(self,patch_size,overlap=100):
        patches = []

        box_corner_threshold = 2

        patch_folder = f"{self.data_folder}/{self.dataset_name}/patches_{patch_size}"
        os.makedirs(patch_folder,exist_ok=True)
        img_patch_folder = f"{patch_folder}/images"
        os.makedirs(img_patch_folder,exist_ok=True)
        label_patch_folder = f"{patch_folder}/labels_original"
        os.makedirs(label_patch_folder,exist_ok=True)
        label_patch_yolo_folder = f"{patch_folder}/labels"
        os.makedirs(label_patch_yolo_folder,exist_ok=True)

        for img_name, img_path in self.img_paths.items():
            # img = cv2.cvtColor(cv2.imread(img_path,1),cv2.COLOR_BGR2RGB)
            img = cv2.imread(img_path,1)
            bbox_coords = self.labels[img_name]['bbox']
            instance_names = self.labels[img_name]['names']
            y_max, x_max, ch = img.shape[:3]
            patch_y, patch_x = self.get_patch_start_coords(my_max=[y_max,x_max],patch_size=patch_size,overlap=overlap)
            for y_0 in patch_y:
                for x_0 in patch_x:
                    my_labels = {'instance_names':[],'rotated_bboxes':[] ,'orthogonal_bboxes':[],'airplane_exist':False}
                    patch = np.zeros(shape=(patch_size,patch_size,ch),dtype=np.uint8)

                    for i,coords in enumerate(bbox_coords):
                        box_corner_in_patch = 0
                        for coord in coords:
                            if (x_0<=coord[0]<=x_0+patch_size) and (y_0<=coord[1]<=y_0+patch_size):
                                box_corner_in_patch += 1
                        if box_corner_in_patch>=box_corner_threshold:
                            # INSTANCE NAMES
                            my_labels['instance_names'].append(instance_names[i])
                            # shift coords
                            shifted_coords = np.array(coords)-[x_0,y_0]
                            # ROTATED BBOXES                     
                            my_labels['rotated_bboxes'].append(shifted_coords.tolist())
                            # ORTHOGONAL BBOXES
                            x_coords = shifted_coords[:,0]
                            y_coords = shifted_coords[:,1]
                            # print(x_coords,y_coords)
                            x_coord_min, x_coord_max = np.amin(x_coords),np.amax(x_coords)
                            y_coord_min, y_coord_max = np.amin(y_coords),np.amax(y_coords)
                            h = y_coord_max - y_coord_min
                            w = x_coord_max - x_coord_min
                            my_labels['orthogonal_bboxes'].append([x_coord_min+w/2.0, y_coord_min+h/2.0, w, h])
                            
                            # AIRPLANE EXIST 
                            my_labels['airplane_exist']=True
                    
                    y_limit_expanded = y_0+patch_size>=y_max
                    x_limit_expanded = x_0+patch_size>=x_max
                    limit_expanded = y_limit_expanded and x_limit_expanded
                    if limit_expanded:
                        patch[:y_max-y_0,:x_max-x_0] = img[y_0:y_max,x_0:x_max]
                    elif y_limit_expanded:
                        patch[:y_max-y_0,:] = img[y_0:y_max,x_0:x_0+patch_size]
                    elif x_limit_expanded:
                        patch[:,:x_max-x_0] = img[y_0:y_0+patch_size,x_0:x_max]                        
                    else:
                        patch[:,:] = img[y_0:y_0+patch_size,x_0:x_0+patch_size]
                    # if my_labels['airplane_exist']:
                        # print(my_labels)
                        # print(my_labels['orthogonal_bboxes'])
                        # self.img_show(img=patch,bboxes=my_labels['rotated_bboxes'],rotated=True)#my_labels)
                        # self.img_show(img=patch,bboxes=my_labels['orthogonal_bboxes'],rotated=False)#my_labels)


                    patch_name = f"{img_name}_x_{x_0}_y_{y_0}"
                    ### save img
                    cv2.imwrite(f"{img_patch_folder}/{patch_name}.png",patch)

                    ### save label
                    with open(f"{label_patch_folder}/{patch_name}.json", 'w') as f:
                        json.dump(my_labels, f,indent=4)

                    ### save label for yolo
                    with open(f"{label_patch_yolo_folder}/{patch_name}.txt", 'w') as f:
                        for bboxes_params in my_labels['orthogonal_bboxes']:
                            params_str = ['0'] # class_id center_x center_y width height
                            for param in bboxes_params:
                                param_norm = param/patch_size
                                if param_norm>1:
                                    param_norm=1
                                elif param_norm<0:
                                    param_norm=0
                                params_str.append(str(param_norm))
                            my_line = ' '.join(params_str)
                            f.write(f"{my_line}\n")
                    # break
        # return patches

    def get_patch_start_coords(self,my_max,patch_size,overlap):

        def get_values(coord_max):
            coords=[]
            coord_0 = 0
            while coord_0 < coord_max:
                coords.append(coord_0)
                coord_0 = coord_0 + patch_size - overlap
                if coord_0+patch_size>coord_max:
                    coords.append(coord_max-patch_size)
                    break
            return coords        
        
        y_max,x_max=my_max

        patch_y = get_values(y_max)
        patch_x = get_values(x_max)

        return patch_y, patch_x
