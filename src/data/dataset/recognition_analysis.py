# class RecognitionAnalysis(Recognition):
#     def __init__(self,dataset_id,dataset_part,dataset_name,patch_size):
#         '''
#         Analyse the recognition data
#         '''
#         super(RecognitionAnalysis, self).__init__(dataset_id,dataset_part,dataset_name,patch_size)
#         self.set_patch_folders()

#     def get_instance_number(self):
#         json_files = os.listdir(self.label_patch_folder)

#         print(f"{self.dataset_part} set of {self.dataset_name} has the following instances:")
#         instance_number = {}
#         for json_file in json_files:
#             patch_dict = json.load(open(f"{self.label_patch_folder}/{json_file}",'r'))
#             instance_name=patch_dict['instance_name']
#             if instance_name not in instance_number.keys():
#                 instance_number[instance_name] = 0
#             else:
#                 instance_number[instance_name] += 1
#         ## SORT DICT
#         instance_number = dict(sorted(instance_number.items()))

#         instance_number['TOTAL'] = sum(instance_number.values())
#         return instance_number
#     def get_airplane_size(self):

#         json_files = os.listdir(self.label_patch_folder)
#         print(f"Airplane size will be calculated for {len(json_files)} planes")
#         size_dict = {}

#         for json_file in json_files:
#            patch_dict = json.load(open(f"{self.label_patch_folder}/{json_file}",'r'))
#            cx,cy,h,w,angle = patch_dict['orthogonal_patch']['bbox_params']
#            instance_name=patch_dict['instance_name']


#            if instance_name not in size_dict.keys():
#                 size_dict[instance_name] = {'w':[],'h':[]}
#            else:
#                 size_dict[instance_name]['w'].append(w)
#                 size_dict[instance_name]['h'].append(h)
#         # print(size_dict)

#         for instance_name in size_dict.keys():
#             total_no = len(size_dict[instance_name]['h'])
#             print(f"{instance_name}: {total_no}")
#             # fig,ax=plt.subplots(2)
#             # fig.suptitle(f'Instance:{instance_name}, total no: {total_no}')
#             # ax[0].set_title('Height')
#             # ax[1].set_title('Width')
#             # ax[0].hist(size_dict[instance_name]['h'],bins=50)
#             # ax[1].hist(size_dict[instance_name]['w'],bins=50)
#             # plt.show()

#     def get_wrong_labels(self,patch_size):
#         json_files = os.listdir(self.label_patch_folder)
#         print(f"Airplane size will be calculated for {len(json_files)} planes")
#         size_dict = {}

#         for json_file in json_files:
#            patch_dict = json.load(open(f"{self.label_patch_folder}/{json_file}",'r'))
#            img = json.load(open(f"{self.img_patch_folder}/{json_file}",'r'))
