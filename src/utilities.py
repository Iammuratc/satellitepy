import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
def show_sample(sample):
    img = sample['image']
    
    # if torch.is_tensor(img):
    #     print('img is torch tensor')
    #     img = img.numpy()
    #     img = img.transpose((1,2,0))#((2, 0, 1))
    
    fig, ax = plt.subplots(1)
    ax.imshow(img,'gray')

    # rotated_bboxes = sample['rotated_bboxes']
    # for coords in rotated_bboxes:
    #     for i, coord in enumerate(coords):
    #         # PLOT BBOX
    #         ax.plot([coords[i-1][0],coord[0]],[coords[i-1][1],coord[1]],c='r')


    orthogonal_bboxes = sample['orthogonal_bboxes']
    for coords in orthogonal_bboxes:
        # x_0 = coords[0][0]
        # y_0 = coords[1][1]
        # w = coords[1][0] - x_0
        # h = coords[1][1] - y_0

        xc,yc,w,h = coords
        rect = patches.Rectangle((xc-w/2.0, yc-h/2.0), w, h, linewidth=1, edgecolor='g', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.show()
    return ax