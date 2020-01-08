import cv2
import numpy as np
import wx
from wx.core import wx
import torch
import params
from ops_data import import_model
from ops_util import brush_stroke_mask
from ops_visual import display_tensor_image


class CanWeFixItGUI(wx.Frame):
    MAX_WIDTH = 256
    MAX_HEIGHT = 256

    def __init__(self, parent, id, title):
        wx.Frame.__init__(self,
                          parent,
                          id,
                          title,
                          wx.DefaultPosition,
                          wx.Size(512, 350))

        self.dirname = ''
        self.filename = ''
        self.img = ''
        self.original_img = None
        self.painted_img = None
        self.mask_img = None
        self.original_img_width = 0
        self.original_img_height = 0
        self.img_width = 0
        self.img_height = 0
        self.drawing = False
        self.prev_point = None
        self.wx_bitmap = None

        self.CreateStatusBar()

        wx.Button(self, 1, 'Inpaint', (10, 130))
        wx.Button(self, 2, 'Load Image', (10, 10))
        wx.Button(self, 3, 'Export Mask', (10, 50))
        wx.Button(self, 4, 'Generate Random Mask', (10, 90))
        self.Bind(wx.EVT_BUTTON, self.onInpaint, id=1)
        self.Bind(wx.EVT_BUTTON, self.onOpenImage, id=2)
        self.Bind(wx.EVT_BUTTON, self.onExportMask, id=3)
        self.Bind(wx.EVT_BUTTON, self.onGenerateRandomMask, id=4)

        self.Centre()
        self.Show()

    def onOpenImage(self, e):
        self.reset_state()

        wildcard = 'JPEG files (*.jpg)|*.jpg|' + \
                   'PNG files (*.png)|*.png|' + \
                   'Other files (*.*)|*.*'
        dlg = wx.FileDialog(self, "Choose an image", wildcard=wildcard, style=wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            self.img = self.dirname + '/' + self.filename

        original_img = cv2.imread(self.img, cv2.IMREAD_COLOR)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        self.original_img_height = original_img.shape[0]
        self.original_img_width = original_img.shape[1]

        if self.original_img_height > self.original_img_width:
            self.img_height = self.MAX_HEIGHT
            self.img_width = int(self.original_img_width * self.MAX_HEIGHT / self.original_img_height)
        else:
            self.img_height = int(self.original_img_height * self.MAX_WIDTH / self.original_img_width)
            self.img_width = self.MAX_WIDTH

        self.original_img = cv2.resize(original_img, (self.img_width, self.img_height))
        self.painted_img = cv2.resize(original_img, (self.img_width, self.img_height))
        self.mask_img = np.zeros((self.img_height, self.img_width, 1), np.uint8)

        wx_img = wx.Bitmap.FromBuffer(self.img_width, self.img_height, self.painted_img)
        self.wx_bitmap = wx.StaticBitmap(self, -1, wx_img, (200, 10))
        self.wx_bitmap.Bind(wx.EVT_LEFT_DOWN, self.on_image_left_down)
        self.wx_bitmap.Bind(wx.EVT_LEFT_UP, self.on_image_left_up)
        self.wx_bitmap.Bind(wx.EVT_MOTION, self.on_image_motion)

        dlg.Destroy()

    def onInpaint(self, e):
        gen = import_model(params.gen_model_path, "G")
        org = torch.from_numpy(self.original_img).type(params.dtype)
        msk = torch.from_numpy(self.mask_img).type(params.dtype)
        img = gen.inpaint_image(org, msk)

        display_tensor_image(img)

    def onExportMask(self, e):
        cv2.imwrite("mask_%s" % self.filename, self.mask_img)

    def onGenerateRandomMask(self, e):
        random_mask = brush_stroke_mask().permute(2, 3, 1, 0).numpy()[:, :, :, -1]
        for i in range(len(random_mask)):
            for j in range(len(random_mask[i])):
                if random_mask[i][j] == 1:
                    random_mask[i][j] = 255

        cv2.imshow("Random Mask", random_mask)

    def on_image_left_down(self, e):
        x, y = e.GetPosition()
        self.drawing = True
        self.prev_point = None

        self.draw_image(x, y)

    def on_image_left_up(self, e):
        self.drawing = False
        self.prev_point = None

    def on_image_motion(self, e):
        if self.drawing:
            x, y = e.GetPosition()
            self.draw_image(x, y)

    def draw_image(self, x, y):
        if self.prev_point is None:
            cv2.circle(self.painted_img, (x, y), 10, (255, 255, 255), -1)
            cv2.circle(self.mask_img, (x, y), 10, (255, 255, 255), -1)
        else:
            cv2.line(self.painted_img, self.prev_point, (x, y), (255, 255, 255), thickness=20)
            cv2.line(self.mask_img, self.prev_point, (x, y), (255, 255, 255), thickness=20)

        self.wx_bitmap.SetBitmap(wx.Bitmap.FromBuffer(self.img_width, self.img_height, self.painted_img))
        self.wx_bitmap.Refresh()
        self.prev_point = (x, y)

    def reset_state(self):
        if self.wx_bitmap:
            self.wx_bitmap.Destroy()
        self.dirname = ''
        self.filename = ''
        self.img = ''
        self.original_img = None
        self.painted_img = None
        self.mask_img = None
        self.original_img_width = 0
        self.original_img_height = 0
        self.img_width = 0
        self.img_height = 0
        self.drawing = False
        self.prev_point = None
        self.wx_bitmap = None


if __name__ == '__main__':
    app = wx.App(False)
    frame = CanWeFixItGUI(None, -1, "Can We Fix It")
    app.MainLoop()
