import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QImage, QPixmap
import pyqtgraph as pg
from scipy.__config__ import show

import vtkmodules.all as vtk
import vtkmodules
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
# from lungmask.thu_paintbrush import path_leaf
import show_coronal_sagittal
import cal_volume_Per_Ven
import overlay_anh
import time
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

import torch
import scipy.ndimage as ndimage
from lungmask import mask
# import ntpath

vtkmodules.qt.QVTKRWIBase = "QGLWidget"


class view_Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        # Set window title and size
        self.setWindowTitle('Inspect.Lung')
        self.setMinimumSize(1550, 950)

        self.setStyleSheet("background-color: #4e456b; padding:5px; color: white")

        # Set central widget and general layout
        self.generalLayout = QGridLayout()
        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.centralWidget.setLayout(self.generalLayout)

        # Create main buttons
        self.guiInit()

        # Load displayed layouts
        self.imageLayout = QStackedLayout()

        self.init_layout = Init_Window()
        self.show_layout = Show_Window()
        self.lobe_per_layout = LOBE_PER_Window()
        self.lobe_ven_layout = LOBE_VEN_Window()
        self.lungSegment_layout = lungSegment_Window()
        self.virtualPlanning_layout = virtualPlanning_Window()

        self.imageLayout.addWidget(self.init_layout)
        self.imageLayout.addWidget(self.show_layout)
        self.imageLayout.addWidget(self.lobe_per_layout)
        self.imageLayout.addWidget(self.lobe_ven_layout)
        self.imageLayout.addWidget(self.lungSegment_layout)
        self.imageLayout.addWidget(self.virtualPlanning_layout)

        self.imageLayout.setCurrentWidget(self.init_layout)
        # self.generalLayout.addLayout(self.imageLayout, 0, 1, 5, 5)
        self.generalLayout.addLayout(self.imageLayout, 0, 0, 5, 4)

        # # Create main buttons
        # self.guiInit()

    def guiInit(self):
        """
            Build layout for Control Buttons
        """

        self.buttonLayout = QVBoxLayout()

        self.buttonLabel = QLabel('Inspect.Lung')
        # self.buttonLabel = QLabel('Main Function')
        self.buttonLabel.setFont(QtGui.QFont('Arial', 20))
        self.buttonLayout.addWidget(self.buttonLabel)

        # NUM OF VOXEL of lobes textbox
        self.volume_show = QLabel()
        self.volume_show.setFixedSize(260, 100)
        self.volume_show.setStyleSheet('background: black')
        self.volume_show.setText(
            'NUMBER OF VOXEL : \n-ULL  : 0 voxel\n-LLL  : 0 voxel\n-URL  : 0 voxel\n-MRL : 0 voxel\n-LRL  : 0 voxel')
        self.buttonLayout.addWidget(self.volume_show)

        # PERCENT OF LOBE
        self.percent_volume = QLabel()
        self.percent_volume.setFixedSize(260, 100)
        self.percent_volume.setStyleSheet('background: black')
        self.percent_volume.setText('VOLUME : \n-ULL  : 0 %\n-LLL  : 0  %\n-URL  : 0  %\n-MRL : 0  %\n-LRL  : 0  %')
        self.buttonLayout.addWidget(self.percent_volume)

        # perfusion of lobes textbox

        self.perfusion_show = QLabel()
        self.perfusion_show.setFixedSize(260, 100)
        self.perfusion_show.setStyleSheet('background: black')
        self.perfusion_show.setText('PERFUSION : \n-ULL  : 0 %\n-LLL  : 0 %\n-URL  : 0 %\n-MRL : 0 %\n-LRL  : 0 %')
        self.buttonLayout.addWidget(self.perfusion_show)

        # ventilation of lobes textbox
        self.ventilation_show = QLabel()
        self.ventilation_show.setFixedSize(260, 100)
        self.ventilation_show.setStyleSheet('background: black')
        self.ventilation_show.setText('VENTILATION : \n-ULL  : 0 %\n-LLL  : 0 %\n-URL  : 0 %\n-MRL : 0 %\n-LRL  : 0 %')
        self.buttonLayout.addWidget(self.ventilation_show)

        self.load_btn = QComboBox()
        self.load_btn.setStyleSheet('QComboBox {text-align:left; font-size:9pt; background-color:rgb(78,69,107)}')
        self.load_btn.setFixedSize(260, 50)
        self.load_btn.addItem('Load Image')
        self.load_btn.addItem('> Load SPECT/CT')
        self.load_btn.addItem('> Load Perfusion')
        self.load_btn.addItem('> Load Ventilation')
        self.buttonLayout.addWidget(self.load_btn)

        self.show_btn = QComboBox()
        self.show_btn.setStyleSheet('QComboBox {text-align:left; font-size:9pt; background-color:rgb(78,69,107)}')
        self.show_btn.setFixedSize(260, 50)
        self.show_btn.addItem('Viewer')
        self.show_btn.addItem('> CT axial')
        self.show_btn.addItem('> CT coronal')
        self.show_btn.addItem('> CT sagittal')
        self.show_btn.addItem('> SPECT_Perfusion_axial')
        self.show_btn.addItem('> SPECT_Perfusion_coronal')
        self.show_btn.addItem('> SPECT_Perfusion_sagittal')
        self.show_btn.addItem('> SPECT_Ventilation_axial')
        self.show_btn.addItem('> SPECT_Ventilation_coronal')
        self.show_btn.addItem('> SPECT_Ventilation_sagittal')
        # self.view_btn.setEnabled(True)
        # self.view_btn.model().item(0).setEnabled(False)
        self.buttonLayout.addWidget(self.show_btn)

        self.segment_btn = QComboBox()
        self.segment_btn.setStyleSheet('QComboBox {text-align:left; font-size:9pt; background-color:rgb(78,69,107)}')
        # self.segment_btn.setStyleSheet('QComboBox {text-align:left; font-size:9pt; background-color:#f0edec}')
        self.segment_btn.setFixedSize(260, 50)
        self.segment_btn.addItem('Segmentation')
        self.segment_btn.addItem('> Lung Segmentation')
        self.segment_btn.addItem('> LOBE-PERFUSION-OVERLAY')
        self.segment_btn.addItem('> LOBE-VENTILATION-OVERLAY')
        self.segment_btn.setEnabled(False)
        self.segment_btn.model().item(0).setEnabled(False)
        self.segment_btn.model().item(1).setEnabled(False)
        self.segment_btn.model().item(2).setEnabled(False)
        self.buttonLayout.addWidget(self.segment_btn)

        self.planning_btn = QComboBox()
        self.planning_btn.setStyleSheet('QComboBox {text-align:left; font-size:9pt; background-color:rgb(78,69,107)}')
        # self.planning_btn.setStyleSheet('QComboBox {text-align:left; font-size:9pt; background-color: #f0edec}')
        self.planning_btn.setFixedSize(260, 50)
        self.planning_btn.addItem('Virtual Planning')
        self.planning_btn.addItem('> Full 3D Render')
        self.planning_btn.setEnabled(False)
        self.planning_btn.model().item(0).setEnabled(False)
        self.buttonLayout.addWidget(self.planning_btn)

        """
            Build layout for confirm/reset button
        """
        self.cmdLayout = QGridLayout()

        self.edit_btn = QPushButton('Edit_Paintbrush')
        self.edit_btn.setStyleSheet(
            'QPushButton {text-align:left; font-size:9pt; padding:5px; background-color:rgb(78,69,107)}')
        # self.edit_btn.setFixedSize(100, 25)
        self.edit_btn.setEnabled(False)
        self.cmdLayout.addWidget(self.edit_btn, 0, 0)

        self.edit_polygon_btn = QPushButton('Edit_Polygon')
        self.edit_polygon_btn.setStyleSheet(
            'QPushButton {text-align:left; font-size:9pt; padding:5px; background-color:rgb(78,69,107)}')
        # self.edit_polygon_btn.setFixedSize(100, 25)
        self.edit_polygon_btn.setEnabled(False)
        self.cmdLayout.addWidget(self.edit_polygon_btn, 0, 1)

        # self.export_btn = QPushButton('Export')
        # self.export_btn.setStyleSheet(
        #     'QPushButton {text-align:left; font-size:9pt; padding:5px; background-color:rgb(78,69,107)}')
        # self.export_btn.setFixedSize(100, 25)
        # self.export_btn.setEnabled(False)
        # self.cmdLayout.addWidget(self.export_btn, 0, 1)

        self.reset_btn = QPushButton('Reset')
        self.reset_btn.setStyleSheet(
            'QPushButton {text-align:left; font-size:9pt; padding:5px;background-color:rgb(78,69,107)}')
        self.cmdLayout.addWidget(self.reset_btn, 1, 0)

        self.run_btn = QPushButton('Run')
        self.run_btn.setStyleSheet(
            'QPushButton {text-align:left; font-size:9pt; padding:5px; background-color:rgb(78,69,107)}')
        self.cmdLayout.addWidget(self.run_btn, 1, 1)

        self.replanning_btn = QPushButton('Refresh Planning')
        self.replanning_btn.setStyleSheet(
            'QPushButton {text-align:center; font-size:9pt; padding:5px; background-color:rgb(78,69,107)}')
        self.replanning_btn.setEnabled(False)
        self.cmdLayout.addWidget(self.replanning_btn, 2, 0, 1, 2)

        # self.save_btn = QPushButton('Save')
        # self.save_btn.setStyleSheet(
        #     'QPushButton {text-align:center; font-size:9pt; padding:5px; background-color:rgb(78,69,107)}')
        # self.save_btn.setEnabled(False)
        # self.cmdLayout.addWidget(self.save_btn, 2, 1)

        self.per_btn = QPushButton('SPECT Perfusion')
        self.per_btn.setStyleSheet(
            'QPushButton {text-align:center; font-size:9pt; padding:2px; background-color:rgb(78,69,107)}')
        self.cmdLayout.addWidget(self.per_btn, 3, 0)

        self.ven_btn = QPushButton('SPECT Ventilation')
        self.ven_btn.setStyleSheet(
            'QPushButton {text-align:center; font-size:9pt; padding:2px; background-color:rgb(78,69,107)}')
        self.cmdLayout.addWidget(self.ven_btn, 3, 1)

        self.cmdLayout.addWidget(QLabel("Overall Label Opacity"), 4, 0)
        self.picker = QDoubleSpinBox()

        self.picker.setMaximum(1)
        self.picker.setMinimum(0)
        self.picker.setSingleStep(0.1)
        self.picker.setValue(0.1)
        self.cmdLayout.addWidget(self.picker, 4, 1)

        self.cmdLayout.addWidget(QLabel("Brush size"), 5, 0)
        self.brush = QDoubleSpinBox()

        self.brush.setMaximum(30)
        self.brush.setMinimum(1)
        self.brush.setSingleStep(1)
        self.brush.setValue(3)
        self.cmdLayout.addWidget(self.brush, 5, 1)

        self.cmdLayout.addWidget(QLabel("Contrast Minimum"), 6, 0)
        self.con_min = QDoubleSpinBox()

        self.con_min.setMaximum(5000)
        self.con_min.setMinimum(-5000)
        self.con_min.setSingleStep(50)
        self.con_min.setValue(-1024)
        self.cmdLayout.addWidget(self.con_min, 6, 1)

        self.cmdLayout.addWidget(QLabel("Contrast Maximum"), 7, 0)
        self.con_max = QDoubleSpinBox()

        self.con_max.setMaximum(5000)
        self.con_max.setMinimum(-5000)
        self.con_max.setSingleStep(50)
        self.con_max.setValue(600)
        self.cmdLayout.addWidget(self.con_max, 7, 1)

        """
            Add to General Layout
        """
        # self.generalLayout.addWidget(self.logo_imv, 0, 4, 1, 0)
        # self.generalLayout.addWidget(self.buttonLabel, 1, 4, 1, 0)
        # self.generalLayout.addLayout(self.buttonLayout, 2, 4, 1, 0)
        self.generalLayout.addLayout(self.buttonLayout, 2, 4)
        self.generalLayout.addLayout(self.cmdLayout, 3, 4)


class control_Window:
    def __init__(self, view_Window, model_Window):
        self.view_Window = view_Window
        self.model_Window = model_Window
        # self.loadcungluc = 1
        self.sizeObject = QDesktopWidget().screenGeometry(-1)
        # print(" Screen size : "  + str(self.sizeObject.height()) + "x"  + str(self.sizeObject.width()))
        self.run_flag = 0
        self.calib_code = cv2.ROTATE_90_CLOCKWISE

        # Preload images

        # PRELOAD = r'/home/avitech-pc-5500/Nam/pre/lung018_isotrop_LVBAC.nii.gz'

        # self.image_path = PRELOAD
        # self.src_SPECT = sitk.ReadImage(self.image_path)
        # self.src_SPECT.SetOrigin(self.src_SPECT.GetOrigin())
        # self.src_SPECT.SetSpacing((self.src_SPECT.GetSpacing()[0]/2,self.src_SPECT.GetSpacing()[1]/2,self.src_SPECT.GetSpacing()[2]/2))
        # self.src_SPECT.SetDirection(self.src_SPECT.GetDirection())
        # self.SP_image = sitk.GetArrayFromImage(self.src_SPECT)
        # self.src = self.src_SPECT

        # # self.imv_SPECT = self.imv_creator(self.image_calib(self.SP_image))
        # # self.view_Window.init_layout.Init_Layout.addWidget(self.imv_SPECT,0,0)

        # self.src_norm = sitk.Cast(sitk.IntensityWindowing(self.src_SPECT, windowMinimum=int(np.min(self.SP_image)),
        #                                         windowMaximum=int(np.max(self.SP_image)),
        #                                         outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)

        # Initialize global variable
        # self.blue = [0, 150, 0] # thuy so 1
        # self.green = [0, 0, 150] # thuy so 2
        # self.yellow = [255, 85, 255] # thuy so 3
        self.glaucous = [0, 255, 255]  # thuy so 4
        # # self.glaucous = [255, 170, 0] # thuy so  4
        # self.red = [255, 170, 0] # thuy so 5
        # self.purple = [78, 69, 107]
        self.blue = [0, 0, 255]
        self.green = [0, 255, 0]
        self.yellow = [255, 255, 0]
        # self.glaucous = [190,185,193]
        self.red = [255, 0, 0]
        self.tan = [210, 180, 140]

        self.lung_pred = np.zeros((128, 256, 256))
        self.lung_overlay = np.zeros((128, 256, 256))
        self.src_lung_label = sitk.Image(256, 256, 128, sitk.sitkUInt8)  # tao anh co kich thuoc

        # Initialize scrollbar and label
        self.scroll_lung_axial = QScrollBar()
        self.scroll_lung_coronal = QScrollBar()
        self.scroll_lung_sagittal = QScrollBar()

        self.scroll_lobe_per_axial = QScrollBar()
        self.scroll_lobe_per_coronal = QScrollBar()
        self.scroll_lobe_per_sagittal = QScrollBar()

        self.scroll_lobe_ven_axial = QScrollBar()
        self.scroll_lobe_ven_coronal = QScrollBar()
        self.scroll_lobe_ven_sagittal = QScrollBar()

        self.label_lung_axial = QLabel()
        self.label_lung_coronal = QLabel()
        self.label_lung_sagittal = QLabel()

        self.label_lobe_per_axial = QLabel()
        self.label_lobe_per_coronal = QLabel()
        self.label_lobe_per_sagittal = QLabel()

        self.label_lobe_ven_axial = QLabel()
        self.label_lobe_ven_coronal = QLabel()
        self.label_lobe_ven_sagittal = QLabel()

        self.lung_lobe_full = QLabel()

        self.ten_lung_axial = QLabel()
        self.ten_lung_coronal = QLabel()
        self.ten_lung_sagittal = QLabel()

        self.imv_lung_axial = ImageView()
        self.imv_lung_coronal = ImageView()
        self.imv_lung_sagittal = ImageView()

        self.imv_lungSegment_axial = ImageView()
        self.imv_lungSegment_coronal = ImageView()
        self.imv_lungSegment_sagittal = ImageView()

        self.imv_Lobe_Per_overlay_axial = ImageView()
        self.imv_Lobe_Per_overlay_coronal = ImageView()
        self.imv_Lobe_Per_overlay_sagittal = ImageView()

        self.imv_Lobe_Ven_overlay_axial = ImageView()
        self.imv_Lobe_Ven_overlay_coronal = ImageView()
        self.imv_Lobe_ven_overlay_sagittal = ImageView()

        self.lung_overlay_calib = np.zeros((128, 256, 256, 3))
        self.lung_overlay_extent_1 = np.zeros((128, 256, 256, 3))
        self.lung_overlay_extent_2 = np.zeros((128, 256, 256, 3))
        self.lung_overlay_extent_ori = np.zeros((128, 256, 256, 3))

        self.SP_overlay = np.zeros((128, 256, 256))
        self.SP_overlay_arr = np.zeros((128, 256, 256))
        self.Per_norm_arr = np.zeros((128, 256, 256, 3))
        self.Per_norm = sitk.Image(256, 256, 128, sitk.sitkUInt8)

        self.slice_lung_axial = 0
        self.slice_lung_coronal = 0
        self.slice_lung_sagittal = 0

        self.slice_lobe_per_axial = 0
        self.slice_lobe_per_coronal = 0
        self.slice_lobe_per_sagittal = 0

        self.slice_lobe_ven_axial = 0
        self.slice_lobe_ven_coronal = 0
        self.slice_lobe_ven_sagittal = 0

        self.lobe_per_overlay_calib = np.zeros((128, 256, 256, 3))
        self.lobe_per_overlay_extent_1 = np.zeros((128, 256, 256, 3))
        self.lobe_per_overlay_extent_2 = np.zeros((128, 256, 256, 3))
        self.lobe_per_overlay_extent_ori = np.zeros((128, 256, 256, 3))

        self.lobe_ven_overlay_calib = np.zeros((128, 256, 256, 3))
        self.lobe_ven_overlay_extent_1 = np.zeros((128, 256, 256, 3))
        self.lobe_ven_overlay_extent_2 = np.zeros((128, 256, 256, 3))
        self.lobe_ven_overlay_extent_ori = np.zeros((128, 256, 256, 3))

        self.lung_ven_flag = 0
        self.lung_per_flag = 0
        self.nhan_brush = 0
        self.nhan_opacity = 0
        self.nhan_contrast_max = 0
        self.nhan_contrast_min = 0
        self.connectSignals()

    def load_Image(self):
        self.view_Window.edit_btn.setEnabled(False)
        # self.view_Window.volume_show.
        self.view_Window.imageLayout.setCurrentWidget(self.view_Window.init_layout)

        option = self.view_Window.load_btn.currentText()
        if option == 'Load Image':
            return

        """ Find image path and load image """
        file_name = QFileDialog.getOpenFileName(
            parent=self.view_Window,
            caption='Open File',
            # directory=r'C:\Users\FR\PycharmProjects\Qtdesigner\Nam',
            directory=os.getcwd(),
            filter='Image files (*.nii.gz *.nii)'
        )

        if file_name == "":
            return
        # self.image_path = file_name[0]

        # self.src = sitk.ReadImage(self.image_path)

        # self.image = sitk.GetArrayFromImage(self.src)

        if option == '> Load SPECT/CT':
            self.image_path = file_name[0]
            self.a = self.path_leaf(self.image_path)
            # print(self.a[0:len(self.a)-7])
            self.src = sitk.ReadImage(self.image_path)

            self.image = sitk.GetArrayFromImage(self.src)

            self.SP_image = self.image
            self.src_SPECT = self.src
            # self.loadcungluc = 1
            self.src_SPECT.SetOrigin(self.src_SPECT.GetOrigin())
            self.src_SPECT.SetSpacing((self.src_SPECT.GetSpacing()[0] / 2, self.src_SPECT.GetSpacing()[1] / 2,
                                       self.src_SPECT.GetSpacing()[2] / 2))
            self.src_SPECT.SetDirection(self.src_SPECT.GetDirection())

            self.imv_SPECT = self.imv_creator(self.image_calib(self.SP_image), levels=(-1200, 200))
            self.view_Window.init_layout.Init_Layout.addWidget(self.imv_SPECT, 0, 0, 8, 8)
            #

            self.SP_image1 = show_coronal_sagittal.coronal(self.image)
            self.SP_image1 = show_coronal_sagittal.coronal_rotate3D(self.SP_image1)
            self.src_SPECT1 = sitk.GetImageFromArray(self.SP_image1)
            self.imv_SPECT1 = self.imv_creator(self.SP_image1, levels=(-1200, 200))
            self.view_Window.init_layout.Init_Layout.addWidget(self.imv_SPECT1, 0, 8, 8, 8)
            #

            self.SP_image2 = show_coronal_sagittal.sagittal(self.image)
            self.SP_image_sag = show_coronal_sagittal.sagittal_rotate3D(self.SP_image2)
            self.src_SPECT2 = sitk.GetImageFromArray(self.SP_image_sag)
            self.imv_SPECT2 = self.imv_creator(self.SP_image_sag, levels=(-1200, 200))
            self.view_Window.init_layout.Init_Layout.addWidget(self.imv_SPECT2, 0, 17, 8, 8)

            self.src_norm = sitk.Cast(sitk.IntensityWindowing(self.src_SPECT, windowMinimum=int(np.min(self.SP_image)),
                                                              windowMaximum=int(np.max(self.SP_image)),
                                                              outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)
        elif option == '> Load Perfusion':

            self.image_path = file_name[0]

            self.src = sitk.ReadImage(self.image_path)

            self.image = sitk.GetArrayFromImage(self.src)

            self.Per_image = self.image  # array
            self.src_Per = self.src  # image
            self.lung_per_flag = 1
            # self.loadcungluc = 1
            # self.src_Per.SetOrigin(self.src_SPECT.GetOrigin())
            # self.src_Per.SetSpacing(self.src_SPECT.GetSpacing())
            # self.src_Per.SetDirection(self.src_SPECT.GetDirection())

            self.imv_Per = self.imv_creator(self.image_calib(self.Per_image))
            self.view_Window.init_layout.Init_Layout.addWidget(self.imv_Per, 8, 0, 8, 8)
            #

            self.Per_image1 = show_coronal_sagittal.coronal(self.image)
            self.Per_image1 = show_coronal_sagittal.coronal_rotate3D(self.Per_image1)
            self.imv_Per1 = self.imv_creator(self.Per_image1)
            self.view_Window.init_layout.Init_Layout.addWidget(self.imv_Per1, 8, 8, 8, 8)
            #

            self.Per_image2 = show_coronal_sagittal.sagittal(self.image)
            self.Per_image_sag = show_coronal_sagittal.sagittal_rotate3D(self.Per_image2)
            self.imv_Per2 = self.imv_creator(self.Per_image_sag)
            self.view_Window.init_layout.Init_Layout.addWidget(self.imv_Per2, 8, 17, 8, 8)

        elif option == '> Load Ventilation':
            self.image_path = file_name[0]

            self.src = sitk.ReadImage(self.image_path)

            self.image = sitk.GetArrayFromImage(self.src)

            self.Ven_image = self.image
            self.src_Ven = self.src
            # self.loadcungluc = 1
            self.lung_ven_flag = 1
            self.src_Ven.SetOrigin(self.src_SPECT.GetOrigin())
            self.src_Ven.SetSpacing(self.src_SPECT.GetSpacing())
            self.src_Ven.SetDirection(self.src_SPECT.GetDirection())

            self.imv_Ven = self.imv_creator(self.image_calib(self.Ven_image))
            self.view_Window.init_layout.Init_Layout.addWidget(self.imv_Ven, 17, 0, 8, 8)
            #

            self.Ven_image1 = show_coronal_sagittal.coronal(self.image)
            self.Ven_image1 = show_coronal_sagittal.coronal_rotate3D(self.Ven_image1)
            self.imv_Ven1 = self.imv_creator(self.Ven_image1)
            self.view_Window.init_layout.Init_Layout.addWidget(self.imv_Ven1, 17, 8, 8, 8)
            #

            self.Ven_image2 = show_coronal_sagittal.sagittal(self.image)
            self.Ven_image2 = show_coronal_sagittal.sagittal_rotate3D(self.Ven_image2)
            self.imv_Ven2 = self.imv_creator(self.Ven_image2)
            self.view_Window.init_layout.Init_Layout.addWidget(self.imv_Ven2, 17, 17, 8, 8)

    def show_overlay_per(self):

        self.view_Window.per_btn.setEnabled(True)
        # overlay perfusion

        self.Per_image = self.image  # array
        self.src_Per = self.src  # image
        self.SP_overlay = overlay_anh.overlay(self.src_SPECT, self.src_Per)
        self.imv_Per = self.imv_creator(self.image_calib(self.SP_overlay))
        self.view_Window.init_layout.Init_Layout.addWidget(self.imv_Per, 8, 0, 8, 8)
        #

        self.src_Per1 = sitk.GetImageFromArray(self.Per_image1)
        self.SP_overlay1 = overlay_anh.overlay(self.src_SPECT1, self.src_Per1)
        self.imv_Per1 = self.imv_creator(self.SP_overlay1)
        self.view_Window.init_layout.Init_Layout.addWidget(self.imv_Per1, 8, 8, 8, 8)
        #

        self.src_Per2 = sitk.GetImageFromArray(self.Per_image_sag)
        self.SP_overlay2 = overlay_anh.overlay(self.src_SPECT2, self.src_Per2)
        self.imv_Per2 = self.imv_creator(self.SP_overlay2)
        self.view_Window.init_layout.Init_Layout.addWidget(self.imv_Per2, 8, 17, 8, 8)

        self.view_Window.imageLayout.setCurrentWidget(self.view_Window.init_layout)

    def show_overlay_ven(self):
        self.view_Window.ven_btn.setEnabled(True)

        # #overlay ventilation
        self.lung_ven_flag = 1

        self.Ven_image = self.image
        self.src_Ven = self.src
        self.SP_overlay3 = overlay_anh.overlay(self.src_SPECT, self.src_Ven)
        self.imv_Ven = self.imv_creator(self.image_calib(self.SP_overlay3))
        self.view_Window.init_layout.Init_Layout.addWidget(self.imv_Ven, 17, 0, 8, 8)

        self.src_Ven1 = sitk.GetImageFromArray(self.Ven_image1)
        self.SP_overlay4 = overlay_anh.overlay(self.src_SPECT1, self.src_Ven1)
        self.imv_Ven1 = self.imv_creator(self.SP_overlay4)
        self.view_Window.init_layout.Init_Layout.addWidget(self.imv_Ven1, 17, 8, 8, 8)
        #     #

        self.src_Ven2 = sitk.GetImageFromArray(self.Ven_image2)
        self.SP_overlay5 = overlay_anh.overlay(self.src_SPECT2, self.src_Ven2)
        self.imv_Ven2 = self.imv_creator(self.SP_overlay5)
        self.view_Window.init_layout.Init_Layout.addWidget(self.imv_Ven2, 17, 17, 8, 8)

        self.view_Window.imageLayout.setCurrentWidget(self.view_Window.init_layout)

    def Show_image(self):

        option = self.view_Window.show_btn.currentText()
        if option == 'Viewer':
            return
        elif option == '> CT axial':
            # self.SP_image = self.image
            self.imv_SPECT = self.imv_creator(self.image_calib(self.SP_image), levels=(-1000, 200))
            self.view_Window.show_layout.Show_Layout.addWidget(self.imv_SPECT, 0, 0)
        elif option == '> CT coronal':
            # self.SP_image1 = show_coronal_sagittal.coronal(self.image)
            # self.SP_image1 = show_coronal_sagittal.coronal_rotate3D(self.SP_image1)
            self.imv_SPECT1 = self.imv_creator(self.SP_image1, levels=(-1000, 200))
            self.view_Window.show_layout.Show_Layout.addWidget(self.imv_SPECT1, 0, 0)
        elif option == '> CT sagittal':
            self.SP_image_sag = show_coronal_sagittal.sagittal_rotate3D(self.SP_image2)
            self.imv_SPECT2 = self.imv_creator(self.SP_image_sag, levels=(-1000, 200))
            self.view_Window.show_layout.Show_Layout.addWidget(self.imv_SPECT2, 0, 0)
        elif option == '> SPECT_Perfusion_axial':
            self.SP_overlay = overlay_anh.overlay(self.src_SPECT, self.src_Per)
            self.imv_Per = self.imv_creator(self.image_calib(self.SP_overlay))
            self.view_Window.show_layout.Show_Layout.addWidget(self.imv_Per, 0, 0)
        elif option == '> SPECT_Perfusion_coronal':
            self.src_Per1 = sitk.GetImageFromArray(self.Per_image1)
            self.SP_overlay1 = overlay_anh.overlay(self.src_SPECT1, self.src_Per1)
            self.imv_Per1 = self.imv_creator(self.SP_overlay1)
            self.view_Window.show_layout.Show_Layout.addWidget(self.imv_Per1, 0, 0)
        elif option == '> SPECT_Perfusion_sagittal':
            self.src_Per2 = sitk.GetImageFromArray(self.Per_image_sag)
            self.SP_overlay2 = overlay_anh.overlay(self.src_SPECT2, self.src_Per2)
            self.imv_Per2 = self.imv_creator(self.SP_overlay2)
            self.view_Window.show_layout.Show_Layout.addWidget(self.imv_Per2, 0, 0)
        elif option == '> SPECT_Ventilation_axial':
            self.SP_overlay3 = overlay_anh.overlay(self.src_SPECT, self.src_Ven)
            self.imv_Ven = self.imv_creator(self.image_calib(self.SP_overlay3))
            self.view_Window.show_layout.Show_Layout.addWidget(self.imv_Ven, 0, 0)
        elif option == '> SPECT_Ventilation_coronal':
            self.src_Ven1 = sitk.GetImageFromArray(self.Ven_image1)
            self.SP_overlay4 = overlay_anh.overlay(self.src_SPECT1, self.src_Ven1)
            self.imv_Ven1 = self.imv_creator(self.SP_overlay4)
            self.view_Window.show_layout.Show_Layout.addWidget(self.imv_Ven1, 0, 0)
        elif option == '> SPECT_Ventilation_sagittal':
            self.src_Ven2 = sitk.GetImageFromArray(self.Ven_image2)
            self.SP_overlay5 = overlay_anh.overlay(self.src_SPECT2, self.src_Ven2)
            self.imv_Ven2 = self.imv_creator(self.SP_overlay5)
            self.view_Window.show_layout.Show_Layout.addWidget(self.imv_Ven2, 0, 0)
        self.view_Window.imageLayout.setCurrentWidget(self.view_Window.show_layout)

    def segment(self):
        option = self.view_Window.segment_btn.currentText()
        self.view_Window.edit_btn.setEnabled(True)
        self.view_Window.edit_polygon_btn.setEnabled(True)

        if option == 'Segmentation':
            return
        elif option == '> Lung Segmentation':
            # tra ve widget hien tai
            self.view_Window.imageLayout.setCurrentWidget(self.view_Window.lungSegment_layout)
        elif option == '> LOBE-PERFUSION-OVERLAY':
            self.view_Window.imageLayout.setCurrentWidget(self.view_Window.lobe_per_layout)
        elif option == '> LOBE-VENTILATION-OVERLAY':
            self.view_Window.imageLayout.setCurrentWidget(self.view_Window.lobe_ven_layout)

    def planning(self):
        self.view_Window.imageLayout.setCurrentWidget(self.view_Window.virtualPlanning_layout)

    def run(self):
        start = time.time()
        # self.run_flag = 1
        if self.run_flag == 1:
            return
        self.run_flag = 1
        self.src = self.src_SPECT
        self.image = sitk.GetArrayFromImage(self.src)

        """ Disable 'Run' Button """
        self.view_Window.run_btn.setEnabled(False)

        """ Confirm load image and disable Load Image option"""
        self.view_Window.load_btn.model().item(1).setEnabled(False)

        """ ------------------------------------- RUN SEGMENTATION ------------------------------------------------------- """
        """
                                                 LUNG SEGMENTATION
        """
        """ Run Lung Segmentation """
        print('Lung Segmentation Running...')
        self.lung_pred, self.src_lung_label= self.model_Window.lungSegment(self.src_SPECT)
        # self.src_lung_label = sitk.ReadImage(r'C:\Users\AVITECH\Downloads\lungmask\lung018_isotrop_LVBAC.roi.nii.gz')
        # self.src_lung_label = sitk.ReadImage(r'/home/avitech-pc-5500/Nam/ve_anh/pre_weight3bo/lung002_CTMAATOMO_CNHAI.nii.gz')
        self.src_lung_label.SetOrigin(self.src_norm.GetOrigin())
        self.src_lung_label.SetSpacing(self.src_norm.GetSpacing())
        self.src_lung_label.SetDirection(self.src_norm.GetDirection())
        self.lung_pred = sitk.GetArrayFromImage(self.src_lung_label)

        import ve_erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        self.src_lung_label_ero = ve_erosion.erosion_five_label(self.src_lung_label, kernel)
        # self.src_lung_label_ero = sitk.GetImageFromArray(self.src_lung_label_ero_arr)
        # self.src_lung_label_ero.SetSpacing(self.src_lung_label.GetSpacing())
        # self.src_lung_label_ero.SetDirection(self.src_lung_label.GetDirection())
        # self.src_lung_label_ero.SetOrigin(self.src_lung_label.GetOrigin())
        import vl

        # print('volume :\n -lobe 1 : {}\n-lobe 2 : {}\n-lobe 3 : {}\n-lobe 4 : {}\n-lobe 5 : {}'.format(self.lobe1, self.lobe2, self.lobe3, self.lobe4, self.lobe5))
        # self.lung_pred, self.src_lung_label= self.model_Window.lungSegment(self.src_SPECT)
        self.lobe1, self.lobe2, self.lobe3, self.lobe4, self.lobe5 = vl.caculate_volume(self.src_lung_label,
                                                                                        self.lung_pred)
        self.view_Window.volume_show.setText(
            'NUMBER OF VOXEL :\n-ULL : {} voxel\n-LLL : {} voxel\n-URL : {} voxel\n-MRL : {} voxel\n-LRL : {} voxel'.format(
                self.lobe1, self.lobe2, self.lobe3, self.lobe4, self.lobe5))

        self.lobe1, self.lobe2, self.lobe3, self.lobe4, self.lobe5 = vl.caculate_percent_volume(self.src_lung_label,
                                                                                                self.lung_pred)
        self.view_Window.percent_volume.setText(
            'VOLUME :\n-ULL : {}  %\n-LLL : {}  %\n-URL : {} %\n-MRL : {}  %\n-LRL : {}  %'.format(self.lobe1,
                                                                                                   self.lobe2,
                                                                                                   self.lobe3,
                                                                                                   self.lobe4,
                                                                                                   self.lobe5))

        if self.lung_per_flag == 1:
            self.per_lobe1, self.per_lobe2, self.per_lobe3, self.per_lobe4, self.per_lobe5 = cal_volume_Per_Ven.cal_percent_Per_Ven(
                self.src_lung_label, self.Per_image)
            self.view_Window.perfusion_show.setText(
                'PERFUSION :\n-ULL : {} %\n-LLL : {} %\n-URL : {} %\n-MRL : {} %\n-LRL : {} %'.format(self.per_lobe1,
                                                                                                      self.per_lobe2,
                                                                                                      self.per_lobe3,
                                                                                                      self.per_lobe4,
                                                                                                      self.per_lobe5))
        if self.lung_ven_flag == 1:
            self.ven_lobe1, self.ven_lobe2, self.ven_lobe3, self.ven_lobe4, self.ven_lobe5 = cal_volume_Per_Ven.cal_percent_Per_Ven(
                self.src_lung_label, self.Ven_image)
            self.view_Window.ventilation_show.setText(
                'VETILATION :\n-ULL : {} %\n-LLL : {} %\n-URL : {} %\n-MRL : {} %\n-LRL : {} %'.format(self.ven_lobe1,
                                                                                                       self.ven_lobe2,
                                                                                                       self.ven_lobe3,
                                                                                                       self.ven_lobe4,
                                                                                                       self.ven_lobe5))
        # chen cac vung mau vao anh
        self.src_lung_overlay = sitk.LabelOverlay(image=self.src_norm, labelImage=self.src_lung_label,
                                                  opacity=0.1, backgroundValue=0.7,
                                                  colormap=self.red + self.green + self.blue + self.yellow + self.glaucous)
        # # chen anh phan doan voi anh per
        # self.src_lung_per_overlay = sitk.LabelOverlay(image=self.src_lung_overlay, labelImage=self.src_Per,
        #                                           opacity=0.1, backgroundValue=0.7,
        #                                           colormap=self.tan)

        self.lung_overlay = sitk.GetArrayFromImage(self.src_lung_overlay)

        #

        # Axial View
        self.imv_lungSegment_axial = self.imv_creator(self.lung_overlay_calib[0], levels=(50, 170))
        self.view_Window.lungSegment_layout.Segment_Layout.addWidget(self.imv_lungSegment_axial, 0, 0)

        # Coronal and Sagittal View
        self.imv_lungSegment_coronal = self.imv_creator(
            cv2.rotate(self.lung_overlay_extent_ori[:, 0, :, :], self.calib_code), levels=(40, 160))
        self.view_Window.lungSegment_layout.Segment_Layout.addWidget(self.imv_lungSegment_coronal, 0, 1)
        self.imv_lungSegment_sagittal = self.imv_creator(
            cv2.rotate(self.lung_overlay_extent_ori[:, :, 0, :], self.calib_code), levels=(40, 160))
        self.view_Window.lungSegment_layout.Segment_Layout.addWidget(self.imv_lungSegment_sagittal, 1, 0)

        # self.imv_Lobeper = self.imv_creator(self.image_calib(self.lung_overlay_calib[0]))
        # self.view_Window.lungSegment_layout.Segment_Layout.addWidget(self.imv_Lobeper, 1, 1)
        """ Create scrollbar and label """
        # self.scroll_lung_axial.setSliderPosition(self.lung_pred.shape[0]//2)
        self.scroll_lung_axial.setSliderPosition(90)
        self.scroll_lung_axial.setRange(0, self.lung_overlay.shape[0] - 1)
        # self.scroll_lung_coronal.setSliderPosition(self.lung_overlay_calib.shape[0])
        self.scroll_lung_coronal.setSliderPosition(90)
        self.scroll_lung_coronal.setRange(0, self.lung_overlay.shape[1] - 1)
        # self.scroll_lung_sagittal.setSliderPosition(self.lung_overlay_calib.shape[0])
        self.scroll_lung_sagittal.setSliderPosition(90)
        self.scroll_lung_sagittal.setRange(0, self.lung_overlay.shape[2] - 1)

        self.label_lung_axial.setStyleSheet('background: black')
        self.label_lung_axial.setText('Lung Segmentation Axial view\nAxial slice: 0')
        self.label_lung_coronal.setStyleSheet('background: black')
        self.label_lung_coronal.setText('Coronal view\nCoronal slice: 90')
        self.label_lung_sagittal.setStyleSheet('background: black')
        self.label_lung_sagittal.setText('Sagittal view\nSagittal slice: 90')

        """ Create and Customize Image Viewer  """
        self.lung_overlay_calib = self.image_calib(self.lung_overlay)
        #

        self.lung_overlay_extent_1 = self.image_extent(self.src_lung_overlay, self.lung_overlay)
        # print(self.lung_overlay_extent_1)
        self.lung_overlay_extent_2 = self.image_extent(self.src_lung_overlay, self.lung_overlay)
        # print(self.lung_overlay_extent_2)
        self.lung_overlay_extent_ori = self.image_extent(self.src_lung_overlay, self.lung_overlay)
        # print(self.lung_overlay_extent_ori)

        """ Arrange Layout """

        self.view_Window.lungSegment_layout.Segment_Layout.addWidget(self.imv_lungSegment_axial, 0, 0, 16, 16)
        self.view_Window.lungSegment_layout.Segment_Layout.addWidget(self.scroll_lung_axial, 0, 16, 17, 1)
        self.view_Window.lungSegment_layout.Segment_Layout.addWidget(self.label_lung_axial, 16, 0, 1, 16)

        self.view_Window.lungSegment_layout.Segment_Layout.addWidget(self.imv_lungSegment_coronal, 0, 17, 7, 7)
        self.view_Window.lungSegment_layout.Segment_Layout.addWidget(self.label_lung_coronal, 7, 17, 1, 7)
        self.view_Window.lungSegment_layout.Segment_Layout.addWidget(self.scroll_lung_coronal, 0, 25, 8, 1)

        self.view_Window.lungSegment_layout.Segment_Layout.addWidget(self.imv_lungSegment_sagittal, 8, 17, 8, 7)
        self.view_Window.lungSegment_layout.Segment_Layout.addWidget(self.label_lung_sagittal, 16, 17, 1, 7)
        self.view_Window.lungSegment_layout.Segment_Layout.addWidget(self.scroll_lung_sagittal, 8, 25, 9, 1)

        # self.view_Window.lungSegment_layout.Segment_Layout.addWidget(self.lung_lobe_full, 36, 0, 40, 20)

        """ Enable 'Lung Segment' Button """
        self.view_Window.segment_btn.setEnabled(True)
        self.view_Window.segment_btn.model().item(0).setEnabled(True)
        self.view_Window.segment_btn.model().item(1).setEnabled(True)

        print('Lung Segmentation Completed')

        """
                                                 LOBE-PERFUSION-OVERLAY
        """
        """ Run LOBE-PERFUSION-OVERLAY """
        if self.lung_per_flag == 1:
            print('LOBE-PERFUSION-OVERLAY Running...')

            self.src_Per.SetOrigin(self.src_lung_label.GetOrigin())
            self.src_Per.SetSpacing(self.src_lung_label.GetSpacing())
            self.src_Per.SetDirection(self.src_lung_label.GetDirection())
            self.src_Lobe_Per_overlay = sitk.LabelOverlay(image=self.src_Per, labelImage=self.src_lung_label_ero,
                                                          opacity=0.1, backgroundValue=0.7,
                                                          colormap=self.red + self.green + self.blue + self.yellow + self.glaucous)
            self.src_Lobe_Per_overlay_arr = sitk.GetArrayFromImage(self.src_Lobe_Per_overlay)

            self.lobe_per_overlay_calib = self.image_calib(self.src_Lobe_Per_overlay_arr)
            self.lobe_per_overlay_extent_1 = self.image_extent(self.src_Lobe_Per_overlay, self.src_Lobe_Per_overlay_arr)
            self.lobe_per_overlay_extent_2 = self.image_extent(self.src_Lobe_Per_overlay, self.src_Lobe_Per_overlay_arr)
            self.lobe_per_overlay_extent_ori = self.image_extent(self.src_Lobe_Per_overlay,
                                                                 self.src_Lobe_Per_overlay_arr)

            self.imv_Lobe_Per_overlay_axial = self.imv_creator(self.lobe_per_overlay_calib[0], levels=(0, 800))
            self.imv_Lobe_Per_overlay_coronal = self.imv_creator(
                cv2.rotate(self.lobe_per_overlay_extent_ori[:, 0, :, :], self.calib_code), levels=(0, 800))
            self.imv_Lobe_Per_overlay_sagittal = self.imv_creator(
                cv2.rotate(self.lobe_per_overlay_extent_ori[:, :, 0, :], self.calib_code), levels=(0, 800))

            """ Create scrollbar and label """
            self.scroll_lobe_per_axial.setSliderPosition(self.lobe_per_overlay_calib.shape[0] // 2)
            self.scroll_lobe_per_axial.setRange(0, self.src_Lobe_Per_overlay_arr.shape[0] - 1)
            self.scroll_lobe_per_coronal.setSliderPosition(self.lobe_per_overlay_calib.shape[0] // 2)
            self.scroll_lobe_per_coronal.setRange(0, self.src_Lobe_Per_overlay_arr.shape[1] - 1)
            self.scroll_lobe_per_sagittal.setSliderPosition(self.lobe_per_overlay_calib.shape[0] // 2)
            self.scroll_lobe_per_sagittal.setRange(0, self.src_Lobe_Per_overlay_arr.shape[2] - 1)

            self.label_lobe_per_axial.setStyleSheet('background-color: black; color: white')
            self.label_lobe_per_axial.setText('Lobe-Perfusion Segmentation Axial view\nAxial slice: 90')
            self.label_lobe_per_coronal.setStyleSheet('background-color: black; color: white')
            self.label_lobe_per_coronal.setText('Coronal view\nCoronal slice: 90')
            self.label_lobe_per_sagittal.setStyleSheet('background-color: black; color: white')
            self.label_lobe_per_sagittal.setText('Sagittal view\nSagittal slice: 90 ')

            """ Arrange Layout """
            self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.imv_Lobe_Per_overlay_axial, 0, 0, 16, 16)
            self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.scroll_lobe_per_axial, 0, 16, 17, 1)
            self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.label_lobe_per_axial, 16, 0, 1, 16)

            self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.imv_Lobe_Per_overlay_coronal, 0, 17, 7, 7)
            self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.label_lobe_per_coronal, 7, 17, 1, 7)
            self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.scroll_lobe_per_coronal, 0, 25, 8, 1)

            self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.imv_Lobe_Per_overlay_sagittal, 8, 17, 8, 7)
            self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.label_lobe_per_sagittal, 16, 17, 1, 7)
            self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.scroll_lobe_per_sagittal, 8, 25, 9, 1)

            """ Enable 'Lung Segment' Button """

            self.view_Window.segment_btn.model().item(2).setEnabled(True)
            print('LOBE-PERFUSION Completed\n')

        """
                                                 LOBE-VENTILATION-OVERLAY
        """
        """ Run LOBE-VENTILATION-OVERLAY """
        if self.lung_ven_flag == 1:
            print('LOBE-VENTILATION-OVERLAY Running...')

            # self.src_Ven.SetDirection(self.src_lung_label.GetDirection())

            self.src_Lobe_Ven_overlay = sitk.LabelOverlay(image=self.src_Ven, labelImage=self.src_lung_label_ero,
                                                          opacity=0.1, backgroundValue=0.7,
                                                          colormap=self.red + self.green + self.blue + self.yellow + self.glaucous)
            self.src_Lobe_Ven_overlay_arr = sitk.GetArrayFromImage(self.src_Lobe_Ven_overlay)

            self.lobe_ven_overlay_calib = self.image_calib(self.src_Lobe_Ven_overlay_arr)
            self.lobe_ven_overlay_extent_1 = self.image_extent(self.src_Lobe_Ven_overlay, self.src_Lobe_Ven_overlay_arr)
            self.lobe_ven_overlay_extent_2 = self.image_extent(self.src_Lobe_Ven_overlay, self.src_Lobe_Ven_overlay_arr)
            self.lobe_ven_overlay_extent_ori = self.image_extent(self.src_Lobe_Ven_overlay,
                                                                 self.src_Lobe_Ven_overlay_arr)

            self.imv_Lobe_Ven_overlay_axial = self.imv_creator(self.lobe_ven_overlay_calib[0], levels=(0, 800))
            self.imv_Lobe_Ven_overlay_coronal = self.imv_creator(
                cv2.rotate(self.lobe_ven_overlay_extent_ori[:, 0, :, :], self.calib_code), levels=(0, 800))
            self.imv_Lobe_Ven_overlay_sagittal = self.imv_creator(
                cv2.rotate(self.lobe_ven_overlay_extent_ori[:, :, 0, :], self.calib_code), levels=(0, 800))

            """ Create scrollbar and label """
            self.scroll_lobe_ven_axial.setSliderPosition(self.lobe_ven_overlay_calib.shape[0] // 2)
            self.scroll_lobe_ven_axial.setRange(0, self.src_Lobe_Ven_overlay_arr.shape[0] - 1)
            self.scroll_lobe_ven_coronal.setSliderPosition(self.lobe_ven_overlay_calib.shape[0] // 2)
            self.scroll_lobe_ven_coronal.setRange(0, self.src_Lobe_Ven_overlay_arr.shape[1] - 1)
            self.scroll_lobe_ven_sagittal.setSliderPosition(self.lobe_ven_overlay_calib.shape[0] // 2)
            self.scroll_lobe_ven_sagittal.setRange(0, self.src_Lobe_Ven_overlay_arr.shape[2] - 1)

            self.label_lobe_ven_axial.setStyleSheet('background-color: black; color: white')
            self.label_lobe_ven_axial.setText('Lobe-Ventilation Segmentation Axial view\nAxial slice: 90')
            self.label_lobe_ven_coronal.setStyleSheet('background-color: black; color: white')
            self.label_lobe_ven_coronal.setText('Coronal view\nCoronal slice: 90')
            self.label_lobe_ven_sagittal.setStyleSheet('background-color: black; color: white')
            self.label_lobe_ven_sagittal.setText('Sagittal view\nSagittal slice: 90 ')

            """ Arrange Layout """
            self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.imv_Lobe_Ven_overlay_axial, 0, 0, 16, 16)
            self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.scroll_lobe_ven_axial, 0, 16, 17, 1)
            self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.label_lobe_ven_axial, 16, 0, 1, 16)

            self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.imv_Lobe_Ven_overlay_coronal, 0, 17, 7, 7)
            self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.label_lobe_ven_coronal, 7, 17, 1, 7)
            self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.scroll_lobe_ven_coronal, 0, 25, 8, 1)

            self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.imv_Lobe_Ven_overlay_sagittal, 8, 17, 8, 7)
            self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.label_lobe_ven_sagittal, 16, 17, 1, 7)
            self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.scroll_lobe_ven_sagittal, 8, 25, 9, 1)

            """ Enable 'Lung Segment' Button """

            self.view_Window.segment_btn.model().item(2).setEnabled(True)
            print('LOBE-VENTILATION Completed\n')

        """ ------------------------------------- RUN VIRTUAL PLANNING ------------------------------------------------------- """
        spacing = np.round_(self.src.GetSpacing()[2]).astype(int)

        self.full_pred_extent = np.zeros(
            (spacing * self.lung_pred.shape[0], self.lung_pred.shape[1], self.lung_pred.shape[2]))
        self.SP_image_extent = np.zeros(
            (spacing * self.lung_pred.shape[0], self.lung_pred.shape[1], self.lung_pred.shape[2]))

        self.SP_image_thresh = np.zeros(self.SP_image.shape)
        for i in range(self.SP_image.shape[0]):
            ret, self.SP_image_thresh[i] = cv2.threshold(np.int16(self.SP_image[i]), 120, 1, cv2.THRESH_BINARY)

        for i in range(self.lung_pred.shape[0]):
            for j in range(spacing):
                if self.calib_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
                    self.full_pred_extent[spacing * i + j] = self.lung_pred[i]
                    self.SP_image_extent[spacing * i + j] = self.SP_image_thresh[i]
                else:
                    self.full_pred_extent[spacing * i + j] = cv2.flip(
                        cv2.rotate(self.lung_pred[i], cv2.ROTATE_90_CLOCKWISE), 1)
                    self.SP_image_extent[spacing * i + j] = cv2.flip(
                        cv2.rotate(self.SP_image_thresh[i], cv2.ROTATE_90_CLOCKWISE), 1)

        self.VTK_Renderer(lung_pred=self.full_pred_extent, img_data=self.SP_image_extent)

        """ Enable 'Virtual Planning' Button """
        self.view_Window.planning_btn.setEnabled(True)
        self.view_Window.planning_btn.model().item(0).setEnabled(True)
        self.view_Window.replanning_btn.setEnabled(True)

        stop = time.time()
        print('Runtime = ' + str(stop - start))
        print('Analysis Complete!')

    def reset(self):
        # os.execl(sys.executable, sys.executable, *sys.argv)
        self.run_flag = 0

        # Reset init layout
        for i in range(1, 10):
            rm = self.view_Window.init_layout.Init_Layout.itemAt(i)
            if rm != None:
                rm.widget().deleteLater()

        self.view_Window.imageLayout.setCurrentWidget(self.view_Window.init_layout)
        self.view_Window.load_btn.model().item(1).setEnabled(True)

        # # Reset segmentations layout
        self.view_Window.segment_btn.setCurrentIndex(0)
        self.view_Window.segment_btn.setEnabled(False)

        # Reset virtual planning layout
        self.view_Window.planning_btn.setCurrentIndex(0)
        self.view_Window.planning_btn.setEnabled(False)
        rm = self.view_Window.virtualPlanning_layout.Planning_Layout.itemAt(0)
        if rm != None:
            rm.widget().deleteLater()

        self.view_Window.run_btn.setEnabled(True)
        self.view_Window.edit_btn.setEnabled(False)
        self.view_Window.replanning_btn.setEnabled(False)
        # os.execl(sys.executable, sys.executable, *sys.argv)

    def update_Opacity(self):
        self.nhan_opacity = 1
        self.opacity = self.view_Window.picker.value()
        self.src_lung_overlay = sitk.LabelOverlay(image=self.src_norm, labelImage=self.src_lung_label,
                                                  opacity=self.opacity, backgroundValue=0.7,
                                                  colormap=self.red + self.green + self.blue + self.yellow + self.glaucous)
        self.lung_overlay = sitk.GetArrayFromImage(self.src_lung_overlay)
        # self.lung_overlay = self.bgr2rgb(self.lung_overlay)

        self.lung_overlay_calib = self.image_calib(self.lung_overlay)
        self.lung_overlay_extent_1 = self.image_extent(self.src_lung_overlay, self.lung_overlay)
        self.lung_overlay_extent_2 = self.image_extent(self.src_lung_overlay, self.lung_overlay)
        self.lung_overlay_extent_ori = self.image_extent(self.src_lung_overlay, self.lung_overlay)

        self.scroll_display(view='axial')

    def update_Brush_size(self):
        # global x
        # self.opacity=  x
        # print(1)
        self.nhan_brush = 1
        self.brush_size = int(self.view_Window.brush.value())

    def update_contrast_min(self):

        self.nhan_contrast_min = 1
        self.min_hist = int(self.view_Window.con_min.value())

    def update_contrast_max(self):

        self.nhan_contrast_max = 1
        self.max_hist = int(self.view_Window.con_max.value())

    def connectSignals(self):
        self.view_Window.load_btn.activated[str].connect(self.load_Image)
        self.view_Window.show_btn.activated[str].connect(self.Show_image)
        self.view_Window.segment_btn.activated[str].connect(self.segment)
        self.view_Window.planning_btn.activated[str].connect(self.planning)

        self.scroll_lung_axial.valueChanged.connect(lambda: self.scroll_display('lung', 'axial'))
        self.scroll_lung_coronal.valueChanged.connect(lambda: self.scroll_display('lung', view='coronal'))
        self.scroll_lung_sagittal.valueChanged.connect(lambda: self.scroll_display('lung', view='sagittal'))

        self.scroll_lobe_per_axial.valueChanged.connect(lambda: self.scroll_display('lobe perfusion', 'axial'))
        self.scroll_lobe_per_coronal.valueChanged.connect(
            lambda: self.scroll_display(segment='lobe perfusion', view='coronal'))
        self.scroll_lobe_per_sagittal.valueChanged.connect(
            lambda: self.scroll_display(segment='lobe perfusion', view='sagittal'))

        self.scroll_lobe_ven_axial.valueChanged.connect(lambda: self.scroll_display('lobe ventilation', 'axial'))
        self.scroll_lobe_ven_coronal.valueChanged.connect(
            lambda: self.scroll_display(segment='lobe ventilation', view='coronal'))
        self.scroll_lobe_ven_sagittal.valueChanged.connect(
            lambda: self.scroll_display(segment='lobe ventilation', view='sagittal'))
        self.view_Window.picker.valueChanged.connect(self.update_Opacity)
        self.view_Window.brush.valueChanged.connect(self.update_Brush_size)
        self.view_Window.con_min.valueChanged.connect(self.update_contrast_min)
        self.view_Window.con_max.valueChanged.connect(self.update_contrast_max)

        self.view_Window.edit_btn.clicked.connect(self.edit_manual)
        self.view_Window.edit_polygon_btn.clicked.connect(self.edit_polygonmode)
        self.view_Window.run_btn.clicked.connect(self.run)
        self.view_Window.per_btn.clicked.connect(self.show_overlay_per)
        self.view_Window.ven_btn.clicked.connect(self.show_overlay_ven)
        self.view_Window.reset_btn.clicked.connect(self.reset)
        self.view_Window.replanning_btn.clicked.connect(self.replanning)
        # self.view_Window.save_btn.clicked.connect(self.save_img)

    def scroll_display(self, segment='lung', view='axial'):
        # n: number of slices
        # h, w, c: height, width, channel
        n, h, w, c = self.lung_overlay_calib.shape
        n1, h1, w1, c1 = self.lung_overlay_extent_ori.shape

        spacing = np.round_(self.src.GetSpacing()[2]).astype(int)
        axial_plane = np.array([0, 512, 0]) * np.ones((h, w, c))
        coronal_plane = np.array([0, 0, 512]) * np.ones((n1, w, c))
        sagittal_plane = np.array([512, 512, 0]) * np.ones((n1, h, c))

        # Scrollbar controller for Lung segmentation
        if segment == 'lung':
            cur_slice_axial = self.scroll_lung_axial.value()
            cur_slice_coronal = self.scroll_lung_coronal.value()
            cur_slice_sagittal = self.scroll_lung_sagittal.value()

            self.lung_axial_hist = self.imv_lungSegment_axial.getHistogramWidget().getLevels()
            self.lung_coronal_hist = self.imv_lungSegment_coronal.getHistogramWidget().getLevels()
            self.lung_sagittal_hist = self.imv_lungSegment_sagittal.getHistogramWidget().getLevels()

            if view == 'axial':
                self.imv_lungSegment_axial.setImage(self.lung_overlay_calib[cur_slice_axial],
                                                    autoRange=False,
                                                    levels=(self.lung_axial_hist[0], self.lung_axial_hist[1]))
                self.label_lung_axial.setText('Lung Segmentation Axial View\nAxial slice: ' + str(cur_slice_axial))

                # Update in coronal and sagitaal view
                self.lung_overlay_extent_1[spacing * self.slice_lung_axial] = self.lung_overlay_extent_ori[
                    spacing * self.slice_lung_axial]
                self.lung_overlay_extent_1[spacing * cur_slice_axial] = axial_plane
                self.lung_overlay_extent_1[:, :, cur_slice_sagittal, :] = sagittal_plane

                self.lung_overlay_extent_2[spacing * self.slice_lung_axial] = self.lung_overlay_extent_ori[
                    spacing * self.slice_lung_axial]
                self.lung_overlay_extent_2[spacing * cur_slice_axial] = axial_plane
                self.lung_overlay_extent_2[:, cur_slice_coronal, :, :] = coronal_plane

                self.imv_lungSegment_coronal.setImage(
                    cv2.rotate(self.lung_overlay_extent_1[:, cur_slice_coronal, :, :], self.calib_code),
                    autoRange=False, levels=(self.lung_coronal_hist[0], self.lung_coronal_hist[1]))
                self.imv_lungSegment_sagittal.setImage(
                    cv2.rotate(self.lung_overlay_extent_2[:, :, cur_slice_sagittal, :], self.calib_code),
                    autoRange=False, levels=(self.lung_sagittal_hist[0], self.lung_sagittal_hist[1]))

                self.slice_lung_axial = cur_slice_axial

            elif view == 'coronal':
                self.imv_lungSegment_coronal.setImage(
                    cv2.rotate(self.lung_overlay_extent_1[:, cur_slice_coronal, :, :], self.calib_code),
                    autoRange=False, levels=(self.lung_coronal_hist[0], self.lung_coronal_hist[1]))
                self.label_lung_coronal.setText('Coronal View\nCoronal slice: ' + str(cur_slice_coronal))

                # Update in sagitaal view
                self.lung_overlay_extent_2[:, self.slice_lung_coronal, :, :] = self.lung_overlay_extent_ori[:,
                                                                               self.slice_lung_coronal, :, :]
                self.lung_overlay_extent_2[:, cur_slice_coronal, :, :] = coronal_plane
                self.lung_overlay_extent_2[spacing * cur_slice_axial] = axial_plane

                self.imv_lungSegment_sagittal.setImage(
                    cv2.rotate(self.lung_overlay_extent_2[:, :, cur_slice_sagittal, :], self.calib_code),
                    autoRange=False, levels=(self.lung_sagittal_hist[0], self.lung_sagittal_hist[1]))

                self.slice_lung_coronal = cur_slice_coronal

            elif view == 'sagittal':
                self.imv_lungSegment_sagittal.setImage(
                    cv2.rotate(self.lung_overlay_extent_2[:, :, cur_slice_sagittal, :], self.calib_code),
                    autoRange=False, levels=(self.lung_sagittal_hist[0], self.lung_sagittal_hist[1]))
                self.label_lung_sagittal.setText('Sagittal View\nSagittal slice: ' + str(cur_slice_sagittal))

                # Update in coronal view
                self.lung_overlay_extent_1[:, :, self.slice_lung_sagittal, :] = self.lung_overlay_extent_ori[:, :,
                                                                                self.slice_lung_sagittal, :]
                self.lung_overlay_extent_1[:, :, cur_slice_sagittal, :] = sagittal_plane
                self.lung_overlay_extent_1[spacing * cur_slice_axial] = axial_plane

                self.imv_lungSegment_coronal.setImage(
                    cv2.rotate(self.lung_overlay_extent_1[:, cur_slice_coronal, :, :], self.calib_code),
                    autoRange=False, levels=(self.lung_coronal_hist[0], self.lung_coronal_hist[1]))

                self.slice_lung_sagittal = cur_slice_sagittal
        # Scrollbar controller for sp window
        elif segment == 'lobe perfusion':
            cur_slice_lobe_per_axial = self.scroll_lobe_per_axial.value()
            cur_slice_lobe_per_coronal = self.scroll_lobe_per_coronal.value()
            cur_slice_lobe_per_sagittal = self.scroll_lobe_per_sagittal.value()

            self.lobe_per_axial_hist = self.imv_Lobe_Per_overlay_axial.getHistogramWidget().getLevels()
            self.lobe_per_coronal_hist = self.imv_Lobe_Per_overlay_coronal.getHistogramWidget().getLevels()
            self.lobe_per_sagittal_hist = self.imv_Lobe_Per_overlay_sagittal.getHistogramWidget().getLevels()

            if view == 'axial':
                self.imv_Lobe_Per_overlay_axial.setImage(self.lobe_per_overlay_calib[cur_slice_lobe_per_axial],
                                                         autoRange=False,
                                                         levels=(
                                                             self.lobe_per_axial_hist[0], self.lobe_per_axial_hist[1]))
                self.label_lobe_per_axial.setText(
                    'Lobe-Perfusion Segmentation Axial View\nAxial slice: ' + str(cur_slice_lobe_per_axial))

                # Update in coronal and sagittal view
                self.lobe_per_overlay_extent_1[spacing * self.slice_lobe_per_axial] = self.lobe_per_overlay_extent_ori[
                    spacing * self.slice_lobe_per_axial]
                self.lobe_per_overlay_extent_1[spacing * cur_slice_lobe_per_axial] = axial_plane
                self.lobe_per_overlay_extent_1[:, :, cur_slice_lobe_per_sagittal, :] = sagittal_plane

                self.lobe_per_overlay_extent_2[spacing * self.slice_lobe_per_axial] = self.lobe_per_overlay_extent_ori[
                    spacing * self.slice_lobe_per_axial]
                self.lobe_per_overlay_extent_2[spacing * cur_slice_lobe_per_axial] = axial_plane
                self.lobe_per_overlay_extent_2[:, cur_slice_lobe_per_coronal, :, :] = coronal_plane

                self.imv_Lobe_Per_overlay_coronal.setImage(
                    cv2.rotate(self.lobe_per_overlay_extent_1[:, cur_slice_lobe_per_coronal, :, :], self.calib_code),
                    autoRange=False, levels=(self.lobe_per_coronal_hist[0], self.lobe_per_coronal_hist[1]))
                self.imv_Lobe_Per_overlay_sagittal.setImage(
                    cv2.rotate(self.lobe_per_overlay_extent_2[:, :, cur_slice_lobe_per_sagittal, :], self.calib_code),
                    autoRange=False, levels=(self.lobe_per_sagittal_hist[0], self.lobe_per_sagittal_hist[1]))

                self.slice_lobe_per_axial = cur_slice_lobe_per_axial

            elif view == 'coronal':
                self.imv_Lobe_Per_overlay_coronal.setImage(
                    cv2.rotate(self.lobe_per_overlay_extent_1[:, cur_slice_lobe_per_coronal, :, :], self.calib_code),
                    autoRange=False, levels=(self.lobe_per_coronal_hist[0], self.lobe_per_coronal_hist[1]))
                self.label_lobe_per_coronal.setText('Coronal View\nCoronal slice: ' + str(cur_slice_lobe_per_coronal))

                # Update in sagittal view
                self.lobe_per_overlay_extent_2[:, self.slice_lobe_per_coronal, :, :] = self.lobe_per_overlay_extent_ori[
                                                                                       :,
                                                                                       self.slice_lobe_per_coronal, :,
                                                                                       :]
                self.lobe_per_overlay_extent_2[:, cur_slice_lobe_per_coronal, :, :] = coronal_plane
                self.lobe_per_overlay_extent_2[spacing * cur_slice_lobe_per_axial] = axial_plane

                self.imv_Lobe_Per_overlay_sagittal.setImage(
                    cv2.rotate(self.lobe_per_overlay_extent_2[:, :, cur_slice_lobe_per_sagittal, :], self.calib_code),
                    autoRange=False, levels=(self.lobe_per_sagittal_hist[0], self.lobe_per_sagittal_hist[1]))

                self.slice_lobe_per_coronal = cur_slice_lobe_per_coronal

            elif view == 'sagittal':
                self.imv_Lobe_Per_overlay_sagittal.setImage(
                    cv2.rotate(self.lobe_per_overlay_extent_2[:, :, cur_slice_lobe_per_sagittal, :], self.calib_code),
                    autoRange=False, levels=(self.lobe_per_sagittal_hist[0], self.lobe_per_sagittal_hist[1]))
                self.label_lobe_per_sagittal.setText(
                    'Sagittal View\nSagittal slice: ' + str(cur_slice_lobe_per_sagittal))

                # Update in coronal view
                self.lobe_per_overlay_extent_1[:, :, self.slice_lobe_per_sagittal,
                :] = self.lobe_per_overlay_extent_ori[:, :,
                     self.slice_lobe_per_sagittal, :]
                self.lobe_per_overlay_extent_1[:, :, cur_slice_lobe_per_sagittal, :] = sagittal_plane
                self.lobe_per_overlay_extent_1[spacing * cur_slice_lobe_per_axial] = axial_plane

                self.imv_Lobe_Per_overlay_coronal.setImage(
                    cv2.rotate(self.lobe_per_overlay_extent_1[:, cur_slice_lobe_per_coronal, :, :], self.calib_code),
                    autoRange=False, levels=(self.lobe_per_coronal_hist[0], self.lobe_per_coronal_hist[1]))

                self.slice_lobe_per_sagittal = cur_slice_lobe_per_sagittal

        # Scrollbar controller for sp window
        elif segment == 'lobe ventilation':
            cur_slice_lobe_ven_axial = self.scroll_lobe_ven_axial.value()
            cur_slice_lobe_ven_coronal = self.scroll_lobe_ven_coronal.value()
            cur_slice_lobe_ven_sagittal = self.scroll_lobe_ven_sagittal.value()

            self.lobe_ven_axial_hist = self.imv_Lobe_Ven_overlay_axial.getHistogramWidget().getLevels()
            self.lobe_ven_coronal_hist = self.imv_Lobe_Ven_overlay_coronal.getHistogramWidget().getLevels()
            self.lobe_ven_sagittal_hist = self.imv_Lobe_Ven_overlay_sagittal.getHistogramWidget().getLevels()

            if view == 'axial':
                self.imv_Lobe_Ven_overlay_axial.setImage(self.lobe_ven_overlay_calib[cur_slice_lobe_ven_axial],
                                                         autoRange=False,
                                                         levels=(
                                                             self.lobe_ven_axial_hist[0], self.lobe_ven_axial_hist[1]))
                self.label_lobe_ven_axial.setText(
                    'Lobe-Ventilation Segmentation Axial View\nAxial slice: ' + str(cur_slice_lobe_ven_axial))

                # Update in coronal and sagittal view
                self.lobe_ven_overlay_extent_1[spacing * self.slice_lobe_ven_axial] = self.lobe_ven_overlay_extent_ori[
                    spacing * self.slice_lobe_ven_axial]
                self.lobe_ven_overlay_extent_1[spacing * cur_slice_lobe_ven_axial] = axial_plane
                self.lobe_ven_overlay_extent_1[:, :, cur_slice_lobe_ven_sagittal, :] = sagittal_plane

                self.lobe_ven_overlay_extent_2[spacing * self.slice_lobe_ven_axial] = self.lobe_ven_overlay_extent_ori[
                    spacing * self.slice_lobe_ven_axial]
                self.lobe_ven_overlay_extent_2[spacing * cur_slice_lobe_ven_axial] = axial_plane
                self.lobe_ven_overlay_extent_2[:, cur_slice_lobe_ven_coronal, :, :] = coronal_plane

                self.imv_Lobe_Ven_overlay_coronal.setImage(
                    cv2.rotate(self.lobe_ven_overlay_extent_1[:, cur_slice_lobe_ven_coronal, :, :], self.calib_code),
                    autoRange=False, levels=(self.lobe_ven_coronal_hist[0], self.lobe_ven_coronal_hist[1]))
                self.imv_Lobe_Ven_overlay_sagittal.setImage(
                    cv2.rotate(self.lobe_ven_overlay_extent_2[:, :, cur_slice_lobe_ven_sagittal, :], self.calib_code),
                    autoRange=False, levels=(self.lobe_ven_sagittal_hist[0], self.lobe_ven_sagittal_hist[1]))

                self.slice_lobe_ven_axial = cur_slice_lobe_ven_axial

            elif view == 'coronal':
                self.imv_Lobe_Ven_overlay_coronal.setImage(
                    cv2.rotate(self.lobe_ven_overlay_extent_1[:, cur_slice_lobe_ven_coronal, :, :], self.calib_code),
                    autoRange=False, levels=(self.lobe_ven_coronal_hist[0], self.lobe_ven_coronal_hist[1]))
                self.label_lobe_ven_coronal.setText('Coronal View\nCoronal slice: ' + str(cur_slice_lobe_ven_coronal))

                # Update in sagittal view
                self.lobe_ven_overlay_extent_2[:, self.slice_lobe_ven_coronal, :, :] = self.lobe_ven_overlay_extent_ori[
                                                                                       :,
                                                                                       self.slice_lobe_ven_coronal, :,
                                                                                       :]
                self.lobe_ven_overlay_extent_2[:, cur_slice_lobe_ven_coronal, :, :] = coronal_plane
                self.lobe_ven_overlay_extent_2[spacing * cur_slice_lobe_ven_axial] = axial_plane

                self.imv_Lobe_Ven_overlay_sagittal.setImage(
                    cv2.rotate(self.lobe_ven_overlay_extent_2[:, :, cur_slice_lobe_ven_sagittal, :], self.calib_code),
                    autoRange=False, levels=(self.lobe_ven_sagittal_hist[0], self.lobe_ven_sagittal_hist[1]))

                self.slice_lobe_ven_coronal = cur_slice_lobe_ven_coronal

            elif view == 'sagittal':
                self.imv_Lobe_Ven_overlay_sagittal.setImage(
                    cv2.rotate(self.lobe_ven_overlay_extent_2[:, :, cur_slice_lobe_ven_sagittal, :], self.calib_code),
                    autoRange=False, levels=(self.lobe_ven_sagittal_hist[0], self.lobe_ven_sagittal_hist[1]))
                self.label_lobe_ven_sagittal.setText(
                    'Sagittal View\nSagittal slice: ' + str(cur_slice_lobe_ven_sagittal))

                # Update in coronal view
                self.lobe_ven_overlay_extent_1[:, :, self.slice_lobe_ven_sagittal,
                :] = self.lobe_ven_overlay_extent_ori[:, :,
                     self.slice_lobe_ven_sagittal, :]
                self.lobe_ven_overlay_extent_1[:, :, cur_slice_lobe_ven_sagittal, :] = sagittal_plane
                self.lobe_ven_overlay_extent_1[spacing * cur_slice_lobe_ven_axial] = axial_plane

                self.imv_Lobe_Ven_overlay_coronal.setImage(
                    cv2.rotate(self.lobe_ven_overlay_extent_1[:, cur_slice_lobe_ven_coronal, :, :], self.calib_code),
                    autoRange=False, levels=(self.lobe_ven_coronal_hist[0], self.lobe_ven_coronal_hist[1]))

                self.slice_lobe_ven_sagittal = cur_slice_lobe_ven_sagittal

    def edit_manual(self):
        self.SP_update_flag = 1
        self.beta = 1
        self.init_flag = 0
        self.init_flag_last_state = 0
        self.exit_flag = 0
        self.ix, self.iy = -1, -1
        self.drawing = False
        self.erasing = False
        # self.brush_size = 3
        # self.edit_color = [0,0,0]
        self.min_hist = -1024
        self.max_hist = 600
        self.lobe_id = 1
        if self.nhan_brush == 1:
            self.brush_size = self.brush_size
        else:
            self.brush_size = 3
        # print(self.brush_size)
        if self.nhan_contrast_min == 1:
            self.min_hist = self.min_hist
        else:
            self.min_hist = -1024

        if self.nhan_contrast_max == 1:
            self.max_hist = self.max_hist
        else:
            self.max_hist = 600

        self.SP_image_edit_manual = self.SP_image.copy()
        self.SP_image_edit_manual[self.SP_image_edit_manual < self.min_hist] = self.min_hist
        self.SP_image_edit_manual[self.SP_image_edit_manual > self.max_hist] = self.max_hist
        self.src_norm_edit_manual = sitk.Cast(sitk.IntensityWindowing(self.src_SPECT,
                                                                      windowMinimum=int(
                                                                          np.min(self.SP_image_edit_manual)),
                                                                      windowMaximum=int(
                                                                          np.max(self.SP_image_edit_manual)),
                                                                      outputMinimum=0.0, outputMaximum=255.0),
                                              sitk.sitkUInt8)

        option = self.view_Window.segment_btn.currentText()
        if option == '> Lung Segmentation':
            # hist_ref = self.imv_NC_lung.getHistogramWidget().getLevels()
            self.edit_color_ori = (self.red + self.green + self.blue + self.yellow + self.glaucous)
            self.edit_color = (self.red + self.green + self.color_bgr2rgb(self.blue) +
                               self.color_bgr2rgb(self.yellow) + self.color_bgr2rgb(self.glaucous))

            self.edit_color_1 = self.green

            self.edit_color_2 = self.blue

            self.edit_color_3 = self.yellow

            self.edit_color_4 = self.glaucous

            self.edit_color_5 = self.red

            self.edit_color_list = [self.green, self.color_bgr2rgb(self.blue), self.color_bgr2rgb(self.yellow),
                                    self.color_bgr2rgb(self.glaucous), self.color_bgr2rgb(self.red)]

            self.overlay = sitk.GetArrayFromImage(
                sitk.LabelOverlay(image=self.src_norm_edit_manual, labelImage=self.src_lung_label,
                                  opacity=0.1, backgroundValue=0, colormap=self.edit_color))

            self.pred_sitk = self.src_lung_label
            self.overlay = self.bgr2rgb(self.overlay)
            self.pred = self.lung_pred.copy()
            self.cur_slice = self.scroll_lung_axial.value()
            mode = 0

        # Prepare original NC image, label image(pred image), display image(overlay image)
        self.pred_color = np.zeros_like(self.overlay)
        for i in range(self.pred_color.shape[0]):
            self.pred_color[i] = cv2.cvtColor(self.pred[i], cv2.COLOR_GRAY2RGB)
        # print(np.unique(self.pred[0]))

        self.pred_color[:, :, :, 0][self.pred_color[:, :, :, 0] == 1] = self.edit_color_1[0]
        self.pred_color[:, :, :, 1][self.pred_color[:, :, :, 1] == 1] = self.edit_color_1[1]
        self.pred_color[:, :, :, 2][self.pred_color[:, :, :, 2] == 1] = self.edit_color_1[2]

        self.pred_color[:, :, :, 0][self.pred_color[:, :, :, 0] == 2] = self.edit_color_5[0]
        self.pred_color[:, :, :, 1][self.pred_color[:, :, :, 1] == 2] = self.edit_color_5[1]
        self.pred_color[:, :, :, 2][self.pred_color[:, :, :, 2] == 2] = self.edit_color_5[2]

        self.pred_color[:, :, :, 0][self.pred_color[:, :, :, 0] == 3] = self.edit_color_4[0]
        self.pred_color[:, :, :, 1][self.pred_color[:, :, :, 1] == 3] = self.edit_color_4[1]
        self.pred_color[:, :, :, 2][self.pred_color[:, :, :, 2] == 3] = self.edit_color_4[2]

        self.pred_color[:, :, :, 0][self.pred_color[:, :, :, 0] == 4] = self.edit_color_3[0]
        self.pred_color[:, :, :, 1][self.pred_color[:, :, :, 1] == 4] = self.edit_color_3[1]
        self.pred_color[:, :, :, 2][self.pred_color[:, :, :, 2] == 4] = self.edit_color_3[2]

        self.pred_color[:, :, :, 0][self.pred_color[:, :, :, 0] == 5] = self.edit_color_2[0]
        self.pred_color[:, :, :, 1][self.pred_color[:, :, :, 1] == 5] = self.edit_color_2[1]
        self.pred_color[:, :, :, 2][self.pred_color[:, :, :, 2] == 5] = self.edit_color_2[2]

        self.pred_ori = self.pred.copy()

        # Change contrast and brightness of reference image
        self.ref = sitk.GetArrayFromImage(
            sitk.LabelOverlay(image=self.src_norm_edit_manual, labelImage=self.src_lung_label,
                              opacity=0, backgroundValue=0, colormap=[1, 1, 1]))
        self.ref_edit = self.ref.copy()

        self.image_edit = self.ref.copy()
        self.mouseMove_buffer = self.image_edit.copy()

        def draw_line(event, x, y, flags, param):
            self.ix, self.iy = x, y
            if (event == cv2.EVENT_MOUSEMOVE and self.init_flag == 1) or self.drawing == True or self.erasing == True:
                if self.drawing == True:
                    cv2.circle(self.pred[self.cur_slice], (self.ix, self.iy), self.brush_size, self.lobe_id, -1)
                    cv2.circle(self.pred_color[self.cur_slice], (self.ix, self.iy), self.brush_size,
                               self.edit_color_list[self.lobe_id - 1], -1)
                    # cv2.circle(self.image_edit[self.cur_slice], (self.ix, self.iy), self.brush_size, self.edit_color,-1)
                    mask = self.pred_color[self.cur_slice].astype(bool)

                    self.image_edit[self.cur_slice][mask] = \
                    cv2.addWeighted(self.ref[self.cur_slice], 0.5, self.pred_color[self.cur_slice], 0.5, 0)[mask]
                    self.mouseMove_buffer[self.cur_slice] = self.image_edit[self.cur_slice].copy()

                elif self.erasing == True:
                    cv2.circle(self.pred[self.cur_slice], (self.ix, self.iy), self.brush_size, 0, -1)
                    cv2.circle(self.pred_color[self.cur_slice], (self.ix, self.iy), self.brush_size, 0, -1)

                    self.image_edit[self.cur_slice][self.pred[self.cur_slice] == 0] = self.ref[self.cur_slice][
                        self.pred[self.cur_slice] == 0]
                    self.mouseMove_buffer[self.cur_slice] = self.image_edit[self.cur_slice].copy()

                else:
                    self.image_edit[self.cur_slice] = self.ref[self.cur_slice].copy()
                    pred_color_clone = self.pred_color[self.cur_slice].copy()
                    cv2.circle(pred_color_clone, (self.ix, self.iy), self.brush_size,
                               self.edit_color_list[self.lobe_id - 1], -1)
                    mask = pred_color_clone.astype(bool)
                    self.image_edit[self.cur_slice][mask] = \
                    cv2.addWeighted(self.ref[self.cur_slice], 0.5, pred_color_clone, 0.5, 0)[mask]
                    # cv2.circle(self.image_edit[self.cur_slice], (self.ix, self.iy), self.brush_size, self.edit_color,-1)

            # left click to draw, right click to erase
            if event == cv2.EVENT_LBUTTONDOWN:
                self.init_flag = 1
                self.drawing = True
                self.erasing = False

            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.erasing = False

                # self.ref_edit = self.ref.copy()
                # contours, hierarchy = cv2.findContours(self.pred[self.cur_slice], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # cv2.drawContours(self.ref_edit[self.cur_slice], contours, -1, 255, 1)
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.init_flag = 1
                self.drawing = False
                self.erasing = True
            elif event == cv2.EVENT_RBUTTONUP:
                self.drawing = False
                self.erasing = False

                # self.ref_edit = self.ref.copy()
                # contours, hierarchy = cv2.findContours(self.pred[self.cur_slice], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # cv2.drawContours(self.ref_edit[self.cur_slice], contours, -1, 255, 1)

            elif event == cv2.EVENT_MOUSEWHEEL:  # cac gia tri am va duong tuong ung cuon ve phia truoc va phia sau tuong ung
                if flags > 0 and self.cur_slice < self.image_edit.shape[0] - 1:
                    # Update slice
                    if self.init_flag == 0:
                        self.image_edit[self.cur_slice + 1] = cv2.convertScaleAbs(self.overlay[self.cur_slice + 1],
                                                                                  beta=self.beta)
                        self.mouseMove_buffer[self.cur_slice + 1] = cv2.convertScaleAbs(
                            self.overlay[self.cur_slice + 1], beta=self.beta)
                        # self.image_edit[self.cur_slice + 1] = self.clipped_zoom(self.overlay[self.cur_slice + 1],2)
                        # self.mouseMove_buffer[self.cur_slice + 1] = self.clipped_zoom(self.overlay[self.cur_slice + 1], 2)

                    self.image_edit[self.cur_slice] = self.mouseMove_buffer[self.cur_slice].copy()
                    self.cur_slice = self.cur_slice + 1

                    self.scroll_lung_axial.setSliderPosition(self.cur_slice)
                    self.scroll_display()

                # Back slide
                elif self.cur_slice != 0:
                    if self.init_flag == 0:
                        self.image_edit[self.cur_slice - 1] = cv2.convertScaleAbs(self.overlay[self.cur_slice - 1],
                                                                                  beta=self.beta)
                        self.mouseMove_buffer[self.cur_slice - 1] = cv2.convertScaleAbs(
                            self.overlay[self.cur_slice - 1], beta=self.beta)
                        # self.image_edit[self.cur_slice - 1] = self.clipped_zoom(self.overlay[self.cur_slice - 1],0.5)
                        # self.mouseMove_buffer[self.cur_slice - 1] =self.clipped_zoom(self.overlay[self.cur_slice - 1], 0.5)

                    self.image_edit[self.cur_slice] = self.mouseMove_buffer[self.cur_slice].copy()
                    self.cur_slice = self.cur_slice - 1

                    self.scroll_lung_axial.setSliderPosition(self.cur_slice)
                    self.scroll_display()

        cv2.namedWindow("edit window", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("reference window", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("edit window", draw_line)

        while True:
            if self.init_flag - self.init_flag_last_state == 1:
                self.init_flag_last_state = 1

                self.SP_image_edit_manual = self.SP_image.copy()
                self.SP_image_edit_manual[self.SP_image_edit_manual < self.min_hist] = self.min_hist
                self.SP_image_edit_manual[self.SP_image_edit_manual > self.max_hist] = self.max_hist
                self.src_norm_edit_manual = sitk.Cast(sitk.IntensityWindowing(self.src_SPECT,
                                                                              windowMinimum=int(
                                                                                  np.min(self.SP_image_edit_manual)),
                                                                              windowMaximum=int(
                                                                                  np.max(self.SP_image_edit_manual)),
                                                                              outputMinimum=0.0, outputMaximum=255.0),
                                                      sitk.sitkUInt8)

                self.ref = sitk.GetArrayFromImage(
                    sitk.LabelOverlay(image=self.src_norm_edit_manual, labelImage=self.src_lung_label,
                                      opacity=0, backgroundValue=0, colormap=[255, 255, 255]))
                self.ref_edit = self.ref.copy()
                # contours, __ = cv2.findContours(self.pred[self.cur_slice], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # cv2.drawContours(self.ref_edit[self.cur_slice], contours, -1, 255, 1)

                for i in range(self.pred.shape[0]):
                    self.image_edit[i] = self.ref[i].copy()
                    mask = self.pred_color[i].astype(bool)
                    self.image_edit[i][mask] = cv2.addWeighted(self.ref[i], 0.5, self.pred_color[i], 0.5, 0)[mask]
                self.mouseMove_buffer = self.image_edit.copy()

            # display the image and wait for a keypress
            # cv2.imshow("edit window", self.image_edit[self.cur_slice])
            # cv2.imshow("reference window", self.ref_edit[self.cur_slice])
            self.image_tron_holizon = np.concatenate((self.image_edit[self.cur_slice], self.ref[self.cur_slice]),
                                                     axis=1)
            cv2.imshow("edit window", self.image_tron_holizon)
            key = cv2.waitKey(1) & 0xFF

            # if the 'exit' key is pressed, break from the loop
            if key == ord("e"):
                print('Exit')
                cv2.destroyAllWindows()
                self.exit_flag = 1
                break



            # Change contrast
            # elif key == ord("w") and self.cur_slice < self.image_edit.shape[0]-1:
            #      # Next slide
            #         # Update slice
            #         if self.init_flag == 0:
            #             self.ref_edit[self.cur_slice+1] = contrast_ls[(self.max_hist - 55)//20][self.cur_slice+1].copy()
            #             self.mouseMove_buffer[self.cur_slice+1] = self.image_edit[self.cur_slice+1].copy()

            #         self.image_edit[self.cur_slice] = self.mouseMove_buffer[self.cur_slice].copy()
            #         self.cur_slice = self.cur_slice + 1
            #         # if mode == 0:
            #         self.scroll_lung_axial.setSliderPosition(self.cur_slice)
            #         self.scroll_display()
            #         # elif mode == 1:
            #         #     self.scroll_liver_axial.setSliderPosition(self.cur_slice)
            #         #     self.scroll_display('liver')
            #         # elif mode == 2:
            #         #     self.scroll_lesion_axial.setSliderPosition(self.cur_slice)
            #         #     self.scroll_display('lesion')

            #         # contours, hierarchy = cv2.findContours(self.pred[self.cur_slice], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #         # cv2.drawContours(self.ref_edit[self.cur_slice], contours, -1, 255, 1)

            # elif key == ord("s") and self.cur_slice != 0:
            #     if self.init_flag == 0:
            #         self.ref_edit[self.cur_slice-1] = contrast_ls[(self.max_hist - 55)//20][self.cur_slice-1].copy()
            #         self.mouseMove_buffer[self.cur_slice-1] = self.image_edit[self.cur_slice-1].copy()

            #     self.image_edit[self.cur_slice] = self.mouseMove_buffer[self.cur_slice].copy()
            #     self.cur_slice = self.cur_slice - 1
            #     # if mode == 0:
            #     self.scroll_lung_axial.setSliderPosition(self.cur_slice)
            #     self.scroll_display()
            #     # elif mode == 1:
            #     #     self.scroll_liver_axial.setSliderPosition(self.cur_slice)
            #     #     self.scroll_display('liver')
            #     # elif mode == 2:
            #     #     self.scroll_lesion_axial.setSliderPosition(self.cur_slice)
            #     #     self.scroll_display('lesion')

            #     # contours, hierarchy = cv2.findContours(self.pred[self.cur_slice], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #     # cv2.drawContours(self.ref_edit[self.cur_slice], contours, -1, 255, 1)

            # Change brush size
            # elif key == ord('a'):
            #     self.brush_size = self.brush_size + 1
            #     self.image_edit[self.cur_slice] = self.mouseMove_buffer[self.cur_slice].copy()
            #     # cv2.circle(self.image_edit[self.cur_slice], (self.ix, self.iy), self.brush_size, self.edit_color,-1)
            #     pred_color_clone = self.pred_color[self.cur_slice].copy()
            #     cv2.circle(pred_color_clone, (self.ix, self.iy), self.brush_size, self.edit_color[self.lobe_id -1], -1)
            #     mask = pred_color_clone.astype(bool)
            #     self.image_edit[self.cur_slice][mask] = cv2.addWeighted(self.ref[self.cur_slice], 0.5, pred_color_clone, 0.5, 0)[mask]

            # elif key == ord('d') and self.brush_size > 0:
            #     self.brush_size = self.brush_size - 1
            #     self.image_edit[self.cur_slice] = self.mouseMove_buffer[self.cur_slice].copy()
            #     # cv2.circle(self.image_edit[self.cur_slice], (self.ix, self.iy), self.brush_size, self.edit_color,-1)
            #     pred_color_clone = self.pred_color[self.cur_slice].copy()
            #     cv2.circle(pred_color_clone, (self.ix, self.iy), self.brush_size, self.edit_color[self.lobe_id -1], -1)
            #     mask = pred_color_clone.astype(bool)
            #     self.image_edit[self.cur_slice][mask] = cv2.addWeighted(self.ref[self.cur_slice], 0.5, pred_color_clone, 0.5, 0)[mask]

            elif key == ord("5"):
                self.lobe_id = 5
                # print(self.edit_color_list[self.lobe_id-1])
            elif key == ord("1"):
                self.lobe_id = 1
                # print(self.edit_color_list[self.lobe_id-1])
            elif key == ord("2"):
                self.lobe_id = 2
                # print(self.edit_color_list[self.lobe_id-1])
            elif key == ord("3"):
                self.lobe_id = 3
                # print(self.edit_color_list[self.lobe_id-1])
            elif key == ord("4"):
                self.lobe_id = 4
                # print(self.edit_color_list[self.lobe_id-1])
            # Add contour
            elif key == ord("c"):
                # Display edited result
                import vl
                import ve_erosion
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
                self.lung_pred = self.pred.copy()
                origin = self.src_lung_label.GetOrigin()
                spacing = self.src_lung_label.GetSpacing()
                direction = self.src_lung_label.GetDirection()

                self.src_lung_label = sitk.GetImageFromArray(self.lung_pred)
                self.src_lung_label.SetOrigin(origin)
                self.src_lung_label.SetSpacing(spacing)
                self.src_lung_label.SetDirection(direction)
                self.src_lung_overlay = sitk.LabelOverlay(image=self.src_norm, labelImage=self.src_lung_label,
                                                          opacity=0.1, backgroundValue=0, colormap=self.edit_color_ori)

                self.lung_overlay = sitk.GetArrayFromImage(self.src_lung_overlay)

                self.lung_overlay_calib = self.image_calib(self.lung_overlay)
                self.lung_overlay_extent_1 = self.image_extent(self.src_lung_overlay, self.lung_overlay)
                self.lung_overlay_extent_2 = self.image_extent(self.src_lung_overlay, self.lung_overlay)
                self.lung_overlay_extent_ori = self.image_extent(self.src_lung_overlay, self.lung_overlay)

                self.scroll_display(view='axial')
                self.lobe1, self.lobe2, self.lobe3, self.lobe4, self.lobe5 = vl.caculate_volume(self.src_lung_label,
                                                                                                self.lung_pred)
                self.view_Window.volume_show.setText(
                    'NUMBER OF VOXEL:\n-ULL : {} voxel\n-LLL : {} voxel\n-URL : {} voxel\n-MRL : {} voxel\n-LRL : {} voxel'.format(
                        self.lobe1, self.lobe2, self.lobe3, self.lobe4, self.lobe5))

                self.lobe1, self.lobe2, self.lobe3, self.lobe4, self.lobe5 = vl.caculate_percent_volume(
                    self.src_lung_label, self.lung_pred)
                self.view_Window.percent_volume.setText(
                    'VOLUME :\n-ULL : {}  %\n-LLL : {}  %\n-URL : {}  %\n-MRL : {}  %\n-LRL : {}  %'.format(self.lobe1,
                                                                                                            self.lobe2,
                                                                                                            self.lobe3,
                                                                                                            self.lobe4,
                                                                                                            self.lobe5))

                if self.lung_per_flag == 1:
                    self.per_lobe1, self.per_lobe2, self.per_lobe3, self.per_lobe4, self.per_lobe5 = cal_volume_Per_Ven.cal_percent_Per_Ven(
                        self.src_lung_label, self.Per_image)
                    self.view_Window.perfusion_show.setText(
                        'PERFUSION :\n-ULL : {} %\n-LLL : {} %\n-URL : {} %\n-MRL : {} %\n-LRL : {} %'.format(
                            self.per_lobe1, self.per_lobe2, self.per_lobe3, self.per_lobe4, self.per_lobe5))
                    # cap nhat lai LOBE-PERFUSION- OVERLAY
                    self.src_lung_label_ero = ve_erosion.erosion_five_label(self.src_lung_label, kernel)
                    self.src_Per.SetOrigin(self.src_lung_label.GetOrigin())
                    self.src_Per.SetSpacing(self.src_lung_label.GetSpacing())
                    self.src_Per.SetDirection(self.src_lung_label.GetDirection())
                    self.src_Lobe_Per_overlay = sitk.LabelOverlay(image=self.src_Per,
                                                                  labelImage=self.src_lung_label_ero,
                                                                  opacity=0.1, backgroundValue=0.7,
                                                                  colormap=self.red + self.green + self.blue + self.yellow + self.glaucous)
                    self.src_Lobe_Per_overlay_arr = sitk.GetArrayFromImage(self.src_Lobe_Per_overlay)

                    self.lobe_per_overlay_calib = self.image_calib(self.src_Lobe_Per_overlay_arr)
                    self.lobe_per_overlay_extent_1 = self.image_extent(self.src_Lobe_Per_overlay,
                                                                       self.src_Lobe_Per_overlay_arr)
                    self.lobe_per_overlay_extent_2 = self.image_extent(self.src_Lobe_Per_overlay,
                                                                       self.src_Lobe_Per_overlay_arr)
                    self.lobe_per_overlay_extent_ori = self.image_extent(self.src_Lobe_Per_overlay,
                                                                         self.src_Lobe_Per_overlay_arr)

                    self.imv_Lobe_Per_overlay_axial = self.imv_creator(self.lobe_per_overlay_calib[0], levels=(0, 800))
                    self.imv_Lobe_Per_overlay_coronal = self.imv_creator(
                        cv2.rotate(self.lobe_per_overlay_extent_ori[:, 0, :, :], self.calib_code), levels=(0, 800))
                    self.imv_Lobe_Per_overlay_sagittal = self.imv_creator(
                        cv2.rotate(self.lobe_per_overlay_extent_ori[:, :, 0, :], self.calib_code), levels=(0, 800))

                    """ Create scrollbar and label """
                    self.scroll_lobe_per_axial.setSliderPosition(self.lobe_per_overlay_calib.shape[0] // 2)
                    self.scroll_lobe_per_axial.setRange(0, self.src_Lobe_Per_overlay_arr.shape[0] - 1)
                    self.scroll_lobe_per_coronal.setSliderPosition(self.lobe_per_overlay_calib.shape[0] // 2)
                    self.scroll_lobe_per_coronal.setRange(0, self.src_Lobe_Per_overlay_arr.shape[1] - 1)
                    self.scroll_lobe_per_sagittal.setSliderPosition(self.lobe_per_overlay_calib.shape[0] // 2)
                    self.scroll_lobe_per_sagittal.setRange(0, self.src_Lobe_Per_overlay_arr.shape[2] - 1)

                    self.label_lobe_per_axial.setStyleSheet('background-color: black; color: white')
                    self.label_lobe_per_axial.setText(' Axial view\nAxial slice: 90')
                    self.label_lobe_per_coronal.setStyleSheet('background-color: black; color: white')
                    self.label_lobe_per_coronal.setText(' Coronal view\nCoronal slice: 90')
                    self.label_lobe_per_sagittal.setStyleSheet('background-color: black; color: white')
                    self.label_lobe_per_sagittal.setText(' Sagittal view\nSagittal slice: 90 ')

                    """ Arrange Layout """
                    self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.imv_Lobe_Per_overlay_axial, 0, 0,
                                                                               16, 16)
                    self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.scroll_lobe_per_axial, 0, 16, 17, 1)
                    self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.label_lobe_per_axial, 16, 0, 1, 16)

                    self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.imv_Lobe_Per_overlay_coronal, 0, 17,
                                                                               7, 7)
                    self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.label_lobe_per_coronal, 7, 17, 1, 7)
                    self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.scroll_lobe_per_coronal, 0, 25, 8,
                                                                               1)

                    self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.imv_Lobe_Per_overlay_sagittal, 8,
                                                                               17, 8, 7)
                    self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.label_lobe_per_sagittal, 16, 17, 1,
                                                                               7)
                    self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.scroll_lobe_per_sagittal, 8, 25, 9,
                                                                               1)

                if self.lung_ven_flag == 1:
                    self.ven_lobe1, self.ven_lobe2, self.ven_lobe3, self.ven_lobe4, self.ven_lobe5 = cal_volume_Per_Ven.cal_percent_Per_Ven(
                        self.src_lung_label, self.Ven_image)
                    self.view_Window.ventilation_show.setText(
                        'VETILATION :\n-ULL : {} %\n-LLL : {} %\n-URL : {} %\n-MRL : {} %\n-LRL : {} %'.format(
                            self.ven_lobe1, self.ven_lobe2, self.ven_lobe3, self.ven_lobe4, self.ven_lobe5))

                    # cap nhat lai LOBE-VENTILATION-OVERLAY
                    self.src_Lobe_Ven_overlay = sitk.LabelOverlay(image=self.src_Ven,
                                                                  labelImage=self.src_lung_label_ero,
                                                                  opacity=0.1, backgroundValue=0.7,
                                                                  colormap=self.red + self.green + self.blue + self.yellow + self.glaucous)
                    self.src_Lobe_Ven_overlay_arr = sitk.GetArrayFromImage(self.src_Lobe_Ven_overlay)

                    self.lobe_ven_overlay_calib = self.image_calib(self.src_Lobe_Ven_overlay_arr)
                    self.lobe_ven_overlay_extent_1 = self.image_extent(self.src_Lobe_Ven_overlay,
                                                                       self.src_Lobe_Ven_overlay_arr)
                    self.lobe_ven_overlay_extent_2 = self.image_extent(self.src_Lobe_Ven_overlay,
                                                                       self.src_Lobe_Ven_overlay_arr)
                    self.lobe_ven_overlay_extent_ori = self.image_extent(self.src_Lobe_Ven_overlay,
                                                                         self.src_Lobe_Ven_overlay_arr)

                    self.imv_Lobe_Ven_overlay_axial = self.imv_creator(self.lobe_ven_overlay_calib[0], levels=(0, 800))
                    self.imv_Lobe_Ven_overlay_coronal = self.imv_creator(
                        cv2.rotate(self.lobe_ven_overlay_extent_ori[:, 0, :, :], self.calib_code), levels=(0, 800))
                    self.imv_Lobe_Ven_overlay_sagittal = self.imv_creator(
                        cv2.rotate(self.lobe_ven_overlay_extent_ori[:, :, 0, :], self.calib_code), levels=(0, 800))

                    """ Create scrollbar and label """
                    self.scroll_lobe_ven_axial.setSliderPosition(self.lobe_ven_overlay_calib.shape[0] // 2)
                    self.scroll_lobe_ven_axial.setRange(0, self.src_Lobe_Ven_overlay_arr.shape[0] - 1)
                    self.scroll_lobe_ven_coronal.setSliderPosition(self.lobe_ven_overlay_calib.shape[0] // 2)
                    self.scroll_lobe_ven_coronal.setRange(0, self.src_Lobe_Ven_overlay_arr.shape[1] - 1)
                    self.scroll_lobe_ven_sagittal.setSliderPosition(self.lobe_ven_overlay_calib.shape[0] // 2)
                    self.scroll_lobe_ven_sagittal.setRange(0, self.src_Lobe_Ven_overlay_arr.shape[2] - 1)

                    self.label_lobe_ven_axial.setStyleSheet('background-color: black; color: white')
                    self.label_lobe_ven_axial.setText(' Axial view\nAxial slice: 90')
                    self.label_lobe_ven_coronal.setStyleSheet('background-color: black; color: white')
                    self.label_lobe_ven_coronal.setText(' Coronal view\nCoronal slice: 90')
                    self.label_lobe_ven_sagittal.setStyleSheet('background-color: black; color: white')
                    self.label_lobe_ven_sagittal.setText(' Sagittal view\nSagittal slice: 90 ')

                    """ Arrange Layout """
                    self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.imv_Lobe_Ven_overlay_axial, 0, 0,
                                                                               16, 16)
                    self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.scroll_lobe_ven_axial, 0, 16, 17, 1)
                    self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.label_lobe_ven_axial, 16, 0, 1, 16)

                    self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.imv_Lobe_Ven_overlay_coronal, 0, 17,
                                                                               7, 7)
                    self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.label_lobe_ven_coronal, 7, 17, 1, 7)
                    self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.scroll_lobe_ven_coronal, 0, 25, 8,
                                                                               1)

                    self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.imv_Lobe_Ven_overlay_sagittal, 8,
                                                                               17, 8, 7)
                    self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.label_lobe_ven_sagittal, 16, 17, 1,
                                                                               7)
                    self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.scroll_lobe_ven_sagittal, 8, 25, 9,
                                                                               1)

        cv2.destroyAllWindows()

    def edit_polygonmode(self):

        # option = self.view_Window.segment_btn.currentText()
        # if option == '> Lung Segmentation':
        #     self.overlay = self.bgr2rgb(self.lung_overlay.copy())
        #     cur_slice = self.scroll_lung_axial.value()
        #     self.pred = self.lung_pred.copy()
        import vl
        import ve_erosion
        self.beta = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        self.min_hist = -1024
        self.max_hist = 600
        self.init_flag = 0  # thiet lap gia tri mouse
        self.init_flag_last_state = 0
        self.contrast_flag = 0
        # self.lobe_id = 1
        # self.opacity = 0.3
        if self.nhan_opacity == 1:
            self.opacity = self.opacity
        else:
            self.opacity = 0.1
        self.SP_image_edit_manual = self.SP_image.copy()
        self.SP_image_edit_manual[self.SP_image_edit_manual < self.min_hist] = self.min_hist
        self.SP_image_edit_manual[self.SP_image_edit_manual > self.max_hist] = self.max_hist
        self.src_norm_edit_manual = sitk.Cast(sitk.IntensityWindowing(self.src_SPECT,
                                                                      windowMinimum=int(
                                                                          np.min(self.SP_image_edit_manual)),
                                                                      windowMaximum=int(
                                                                          np.max(self.SP_image_edit_manual)),
                                                                      outputMinimum=0.0, outputMaximum=255.0),
                                              sitk.sitkUInt8)

        self.lobe_stt = 1
        self.edit_color_ori = (self.red + self.green + self.blue + self.yellow + self.glaucous)
        self.edit_color = (self.color_bgr2rgb(self.red) + self.green + self.color_bgr2rgb(self.blue) +
                           self.color_bgr2rgb(self.yellow) + self.color_bgr2rgb(self.glaucous))

        self.edit_color_list = [self.green, self.color_bgr2rgb(self.blue), self.color_bgr2rgb(self.yellow),
                                self.color_bgr2rgb(self.glaucous), self.color_bgr2rgb(self.red)]

        self.overlay = sitk.GetArrayFromImage(
            sitk.LabelOverlay(image=self.src_norm_edit_manual, labelImage=self.src_lung_label,
                              opacity=self.opacity, backgroundValue=0.7, colormap=self.edit_color))

        self.pred = self.lung_pred.copy()
        self.pred_ori = self.pred.copy()
        self.cur_slice = self.scroll_lung_axial.value()

        self.image_edit = self.overlay.copy()

        # Each slice has a list of annotation
        self.annotation = []
        for i in range(self.image_edit.shape[0]):
            self.annotation.append([])

        self.cur_annotation = self.annotation[self.cur_slice]
        # self.image_edit = self.image_3d[cur_slice]
        self.clone = self.overlay[self.cur_slice].copy()

        self.refPt = []
        self.count = 0
        self.dagiac = 0

        def list2np(ls):
            len_list = len(ls)
            np_arr = np.zeros((len_list, 1, 2))
            for i in range(len(ls)):
                np_arr[i] = ls[i]

            return np_arr

        def draw_line(event, x, y, flags, param):
            # left click for drawing the next lines
            if event == cv2.EVENT_LBUTTONDOWN:
                self.init_flag = 1
                self.refPt.append((x, y))
                cv2.line(self.pred[self.cur_slice], self.refPt[self.count - 1], self.refPt[self.count], (0, 0, 0), 1)
                cv2.line(self.image_edit[self.cur_slice], self.refPt[self.count - 1], self.refPt[self.count], (0, 0, 0),
                         1)
                self.count = self.count + 1

            # right click for deleting the previous line
            elif event == cv2.EVENT_RBUTTONDOWN:
                # if current point is the beginning of a new contour/the end of current contour
                if self.count == 0 and len(self.cur_annotation) != 0:
                    # If the current point is the end of the current contour -> delete current contour
                    if len(self.cur_annotation[len(self.cur_annotation) - 1]) == 0:
                        self.cur_annotation.remove(self.cur_annotation[len(self.cur_annotation) - 1])

                    self.refPt = self.cur_annotation[len(self.cur_annotation) - 1]
                    self.count = len(self.refPt)
                    self.cur_annotation.remove(self.cur_annotation[len(self.cur_annotation) - 1])

                self.image_edit[self.cur_slice] = self.clone.copy()
                for i in range(len(self.cur_annotation)):
                    refPt_local = self.cur_annotation[i]
                    count1 = 0
                    for j in range(len(refPt_local)):
                        cv2.line(self.image_edit[self.cur_slice], refPt_local[count1 - 1], refPt_local[count1],
                                 (0, 0, 0), 1)
                        count1 = count1 + 1

                del self.refPt[self.count - 1]
                self.count = self.count - 1
                for i in range(1, self.count):
                    cv2.line(self.image_edit[self.cur_slice], self.refPt[i - 1], self.refPt[i], (0, 0, 0), 1)

            elif event == cv2.EVENT_MOUSEWHEEL:
                if self.cur_slice != 0:
                    self.image_edit[self.cur_slice - 1] = cv2.convertScaleAbs(self.overlay[self.cur_slice - 1],
                                                                              beta=self.beta)

                    self.cur_slice = self.cur_slice - 1
                    self.scroll_lung_axial.setSliderPosition(self.cur_slice)
                    self.scroll_display()

        cv2.namedWindow("image_editor", cv2.WINDOW_NORMAL)

        cv2.setMouseCallback("image_editor", draw_line)

        while True:

            cv2.imshow("image_editor", self.image_edit[self.cur_slice])
            key = cv2.waitKey(1) & 0xFF
            # if the 'reset' key is pressed, reset all
            if key == ord("r"):
                self.image_edit[self.cur_slice] = self.clone.copy()
                self.pred[self.cur_slice] = self.pred_ori[self.cur_slice].copy()
            # mark the end of the current contour
            elif key == ord(" "):
                if len(self.refPt) != 0:
                    # add the 1st point to the last position and complete the current contour
                    self.refPt.append(self.refPt[0])
                    cv2.line(self.image_edit[self.cur_slice], self.refPt[self.count - 1], self.refPt[self.count],
                             (0, 0, 0), 1)

                    # add current contour to current annotation
                    self.cur_annotation.append(self.refPt)
                    # reset refPt and count
                    self.refPt = []
                    self.count = 0

            # next slice
            elif key == ord("w") and self.cur_slice < self.image_edit.shape[0] - 1:
                # Update current annotation to the current slice
                self.cur_annotation.append(self.refPt)
                # self.image_edit[self.cur_slice] = self.image_edit
                # Update slice
                self.cur_slice = self.cur_slice + 1
                # Update current annotation, clone(cache), image_edit
                # self.image_edit = self.image_edit[self.cur_slice]
                self.cur_annotation = self.annotation[self.cur_slice]
                self.clone = self.overlay[self.cur_slice].copy()
                # Continue from the latest state
                if len(self.cur_annotation) == 0:
                    # if the slice is original, initialize refPt and count
                    self.refPt = []
                    self.count = 0
                else:
                    # if slice has already been modified, continue from the latest state
                    self.refPt = self.cur_annotation[len(self.cur_annotation) - 1]
                    self.count = len(self.refPt)
                    self.cur_annotation.remove(self.cur_annotation[len(self.cur_annotation) - 1])


            # back slice
            elif key == ord("s") and self.cur_slice != 0:

                self.cur_annotation.append(self.refPt)
                # self.image_edit[self.cur_slice] = self.image_edit
                # Update slice
                self.cur_slice = self.cur_slice - 1
                # Update current annotation, clone(cache), image_edit
                # self.image_edit = self.image_edit[self.cur_slice]
                self.cur_annotation = self.annotation[self.cur_slice]
                self.clone = self.overlay[self.cur_slice].copy()
                # Continue from the latest state
                if len(self.cur_annotation) == 0:
                    # if the slice is original, initialize refPt and count
                    self.refPt = []
                    self.count = 0
                else:
                    # if slice has already been modified, continue from the latest state
                    self.refPt = self.cur_annotation[len(self.cur_annotation) - 1]
                    self.count = len(self.refPt)
                    self.cur_annotation.remove(self.cur_annotation[len(self.cur_annotation) - 1])


            elif key == ord("a") or key == ord("d"):
                key_past = key
                while (1):
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("1") or key == ord("2") or key == ord("3") or key == ord("4") or key == ord("5"):
                        break

                for i in range(len(self.annotation)):
                    # print(len(self.annotation))
                    self.cur_annotation = self.annotation[i]
                    if len(self.cur_annotation) != 0 and len(self.cur_annotation[0]) != 0:
                        # print(len(self.cur_annotation))
                        for j in range(len(self.cur_annotation)):
                            draw = self.cur_annotation[j]
                            if len(draw) != 0 and key_past == ord("a"):
                                if key == ord("1"):
                                    # cv2.drawContours(self.lung_pred[i], [list2np(draw).astype(int)], 0, 1, -1)
                                    cv2.drawContours(self.pred[i], [list2np(draw).astype(int)], 0, 1, -1)
                                    # doan nay them
                                    self.lung_pred = self.pred.copy()
                                    origin = self.src_lung_label.GetOrigin()
                                    spacing = self.src_lung_label.GetSpacing()
                                    direction = self.src_lung_label.GetDirection()

                                    self.src_lung_label = sitk.GetImageFromArray(self.lung_pred)
                                    self.src_lung_label.SetOrigin(origin)
                                    self.src_lung_label.SetSpacing(spacing)
                                    self.src_lung_label.SetDirection(direction)
                                    self.src_lung_overlay = sitk.LabelOverlay(image=self.src_norm_edit_manual,
                                                                              labelImage=self.src_lung_label,
                                                                              opacity=self.opacity, backgroundValue=0,
                                                                              colormap=self.edit_color)

                                    self.lung_edit_overlay = sitk.GetArrayFromImage(self.src_lung_overlay)
                                    self.image_edit = self.lung_edit_overlay.copy()
                                    # KET thuc them
                                    # cv2.drawContours(self.image_edit[i], [list2np(draw).astype(int)], 0, self.edit_color_list[0], -1)

                                    # break
                                elif key == ord("2"):
                                    # cv2.drawContours(self.lung_pred[i], [list2np(draw).astype(int)], 0, 2, -1)
                                    cv2.drawContours(self.pred[i], [list2np(draw).astype(int)], 0, 2, -1)
                                    # doan nay them
                                    self.lung_pred = self.pred.copy()
                                    origin = self.src_lung_label.GetOrigin()
                                    spacing = self.src_lung_label.GetSpacing()
                                    direction = self.src_lung_label.GetDirection()

                                    self.src_lung_label = sitk.GetImageFromArray(self.lung_pred)
                                    self.src_lung_label.SetOrigin(origin)
                                    self.src_lung_label.SetSpacing(spacing)
                                    self.src_lung_label.SetDirection(direction)
                                    self.src_lung_overlay = sitk.LabelOverlay(image=self.src_norm_edit_manual,
                                                                              labelImage=self.src_lung_label,
                                                                              opacity=self.opacity, backgroundValue=0,
                                                                              colormap=self.edit_color)

                                    self.lung_edit_overlay = sitk.GetArrayFromImage(self.src_lung_overlay)
                                    self.image_edit = self.lung_edit_overlay.copy()
                                    # KET thuc them
                                    # cv2.drawContours(self.image_edit[i], [list2np(draw).astype(int)], 0, self.edit_color_list[1], -1)

                                    # break
                                elif key == ord("3"):
                                    # cv2.drawContours(self.lung_pred[i], [list2np(draw).astype(int)], 0, 3, -1)
                                    cv2.drawContours(self.pred[i], [list2np(draw).astype(int)], 0, 3, -1)
                                    # doan nay them
                                    self.lung_pred = self.pred.copy()
                                    origin = self.src_lung_label.GetOrigin()
                                    spacing = self.src_lung_label.GetSpacing()
                                    direction = self.src_lung_label.GetDirection()

                                    self.src_lung_label = sitk.GetImageFromArray(self.lung_pred)
                                    self.src_lung_label.SetOrigin(origin)
                                    self.src_lung_label.SetSpacing(spacing)
                                    self.src_lung_label.SetDirection(direction)
                                    self.src_lung_overlay = sitk.LabelOverlay(image=self.src_norm_edit_manual,
                                                                              labelImage=self.src_lung_label,
                                                                              opacity=self.opacity, backgroundValue=0,
                                                                              colormap=self.edit_color)

                                    self.lung_edit_overlay = sitk.GetArrayFromImage(self.src_lung_overlay)
                                    self.image_edit = self.lung_edit_overlay.copy()
                                    # KET thuc them
                                    # cv2.drawContours(self.image_edit[i], [list2np(draw).astype(int)], 0, self.edit_color_list[2], -1)

                                    # break
                                elif key == ord("4"):
                                    # cv2.drawContours(self.lung_pred[i], [list2np(draw).astype(int)], 0, 4, -1)
                                    cv2.drawContours(self.pred[i], [list2np(draw).astype(int)], 0, 4, -1)
                                    # doan nay them
                                    self.lung_pred = self.pred.copy()
                                    origin = self.src_lung_label.GetOrigin()
                                    spacing = self.src_lung_label.GetSpacing()
                                    direction = self.src_lung_label.GetDirection()

                                    self.src_lung_label = sitk.GetImageFromArray(self.lung_pred)
                                    self.src_lung_label.SetOrigin(origin)
                                    self.src_lung_label.SetSpacing(spacing)
                                    self.src_lung_label.SetDirection(direction)
                                    self.src_lung_overlay = sitk.LabelOverlay(image=self.src_norm_edit_manual,
                                                                              labelImage=self.src_lung_label,
                                                                              opacity=self.opacity, backgroundValue=0,
                                                                              colormap=self.edit_color)

                                    self.lung_edit_overlay = sitk.GetArrayFromImage(self.src_lung_overlay)
                                    self.image_edit = self.lung_edit_overlay.copy()
                                    # KET thuc them
                                    # cv2.drawContours(self.image_edit[i], [list2np(draw).astype(int)], 0,self.edit_color_list[3], -1)

                                    # break
                                elif key == ord("5"):
                                    # cv2.drawContours(self.lung_pred[i], [list2np(draw).astype(int)], 0, 5, -1)

                                    cv2.drawContours(self.pred[i], [list2np(draw).astype(int)], 0, 5, -1)
                                    # doan nay them
                                    self.lung_pred = self.pred.copy()
                                    origin = self.src_lung_label.GetOrigin()
                                    spacing = self.src_lung_label.GetSpacing()
                                    direction = self.src_lung_label.GetDirection()

                                    self.src_lung_label = sitk.GetImageFromArray(self.lung_pred)
                                    self.src_lung_label.SetOrigin(origin)
                                    self.src_lung_label.SetSpacing(spacing)
                                    self.src_lung_label.SetDirection(direction)
                                    self.src_lung_overlay = sitk.LabelOverlay(image=self.src_norm_edit_manual,
                                                                              labelImage=self.src_lung_label,
                                                                              opacity=self.opacity, backgroundValue=0,
                                                                              colormap=self.edit_color)

                                    self.lung_edit_overlay = sitk.GetArrayFromImage(self.src_lung_overlay)
                                    self.image_edit = self.lung_edit_overlay.copy()
                                    # KET thuc them
                                    # cv2.drawContours(self.image_edit[i], [list2np(draw).astype(int)], 0, self.edit_color_list[4], -1)

                            elif len(draw) != 0 and key_past == ord("d"):
                                cv2.drawContours(self.pred[i], [list2np(draw).astype(int)], 0, 0, -1)
                                # doan nay them
                                self.lung_pred = self.pred.copy()
                                origin = self.src_lung_label.GetOrigin()
                                spacing = self.src_lung_label.GetSpacing()
                                direction = self.src_lung_label.GetDirection()

                                self.src_lung_label = sitk.GetImageFromArray(self.lung_pred)
                                self.src_lung_label.SetOrigin(origin)
                                self.src_lung_label.SetSpacing(spacing)
                                self.src_lung_label.SetDirection(direction)
                                self.src_lung_overlay = sitk.LabelOverlay(image=self.src_norm_edit_manual,
                                                                          labelImage=self.src_lung_label,
                                                                          opacity=self.opacity, backgroundValue=0,
                                                                          colormap=self.edit_color)

                                self.lung_edit_overlay = sitk.GetArrayFromImage(self.src_lung_overlay)
                                self.image_edit = self.lung_edit_overlay.copy()
                                # KET thuc them
                                # cv2.drawContours(self.image_edit[i], [list2np(draw).astype(int)], 0, 0, -1)

            elif key == ord("z"):
                self.opacity += 0.02
                self.contrast_flag = 1

            elif key == ord("x"):
                self.opacity -= 0.02
                self.contrast_flag = 1
            elif key == ord("6"):
                self.min_hist -= 8
                self.contrast_flag = 1

            elif key == ord("7") and self.min_hist < self.max_hist:
                self.min_hist += 8
                self.contrast_flag = 1

            elif key == ord("8") and self.max_hist > self.min_hist:
                self.max_hist -= 8
                self.contrast_flag = 1

            elif key == ord("9"):
                self.max_hist += 8
                self.contrast_flag = 1


            elif key == ord("c"):
                # Display edited result
                # save = '/home/avitech-pc-5500/Nam/lungmask/luuanhve'
                self.lung_pred = self.pred.copy()
                origin = self.src_lung_label.GetOrigin()
                spacing = self.src_lung_label.GetSpacing()
                direction = self.src_lung_label.GetDirection()

                self.src_lung_label = sitk.GetImageFromArray(self.lung_pred)
                self.src_lung_label.SetOrigin(origin)
                self.src_lung_label.SetSpacing(spacing)
                self.src_lung_label.SetDirection(direction)
                self.src_lung_overlay = sitk.LabelOverlay(image=self.src_norm, labelImage=self.src_lung_label,
                                                          opacity=self.opacity, backgroundValue=0.7,
                                                          colormap=self.edit_color_ori)

                # save = QFileDialog.getExistingDirectory(caption='Chọn file bạn muốn lưu')

                # sitk.WriteImage(self.src_lung_label, fileName= os.path.join(save + '/' +self.a.split('.')[0] + ".roi.nii.gz")  , useCompression=True)
                # name = QFileDialog.getSaveFileName(
                #     parent=self.view_Window,
                #     caption='SaveFile',

                #     directory=os.getcwd(),
                #     filter='Image files (*.nii.gz *.nii)'

                # )
                # os.getcwd()
                # file = open(name,'w')
                # text = self.textEdit.toPlainText()
                # file.write(text)
                # file.close()
                self.lung_overlay = sitk.GetArrayFromImage(self.src_lung_overlay)

                self.lung_overlay_calib = self.image_calib(self.lung_overlay)
                self.lung_overlay_extent_1 = self.image_extent(self.src_lung_overlay, self.lung_overlay)
                self.lung_overlay_extent_2 = self.image_extent(self.src_lung_overlay, self.lung_overlay)
                self.lung_overlay_extent_ori = self.image_extent(self.src_lung_overlay, self.lung_overlay)

                self.scroll_display(view='axial')

                self.lobe1, self.lobe2, self.lobe3, self.lobe4, self.lobe5 = vl.caculate_volume(self.src_lung_label,
                                                                                                self.lung_pred)
                self.view_Window.volume_show.setText(
                    'NUMBER OF VOXEL:\n-ULL : {} voxel\n-LLL : {} voxel\n-URL : {} voxel\n-MRL : {} voxel\n-LRL : {} voxel'.format(
                        self.lobe1, self.lobe2, self.lobe3, self.lobe4, self.lobe5))

                self.lobe1, self.lobe2, self.lobe3, self.lobe4, self.lobe5 = vl.caculate_percent_volume(
                    self.src_lung_label, self.lung_pred)
                self.view_Window.percent_volume.setText(
                    'VOLUME :\n-ULL : {}  %\n-LLL : {}  %\n-URL : {}  %\n-MRL : {}  %\n-LRL : {}  %'.format(self.lobe1,
                                                                                                            self.lobe2,
                                                                                                            self.lobe3,
                                                                                                            self.lobe4,
                                                                                                            self.lobe5))

                if self.lung_per_flag == 1:
                    self.per_lobe1, self.per_lobe2, self.per_lobe3, self.per_lobe4, self.per_lobe5 = cal_volume_Per_Ven.cal_percent_Per_Ven(
                        self.src_lung_label, self.Per_image)
                    self.view_Window.perfusion_show.setText(
                        'PERFUSION :\n-ULL : {} %\n-LLL : {} %\n-URL : {} %\n-MRL : {} %\n-LRL : {} %'.format(
                            self.per_lobe1, self.per_lobe2, self.per_lobe3, self.per_lobe4, self.per_lobe5))
                    # cap nhat lai LOBE-PERFUSION- OVERLAY
                    self.src_lung_label_ero = ve_erosion.erosion_five_label(self.src_lung_label, kernel)
                    self.src_Per.SetOrigin(self.src_lung_label.GetOrigin())
                    self.src_Per.SetSpacing(self.src_lung_label.GetSpacing())
                    self.src_Per.SetDirection(self.src_lung_label.GetDirection())
                    self.src_Lobe_Per_overlay = sitk.LabelOverlay(image=self.src_Per,
                                                                  labelImage=self.src_lung_label_ero,
                                                                  opacity=0.1, backgroundValue=0.7,
                                                                  colormap=self.red + self.green + self.blue + self.yellow + self.glaucous)
                    self.src_Lobe_Per_overlay_arr = sitk.GetArrayFromImage(self.src_Lobe_Per_overlay)

                    self.lobe_per_overlay_calib = self.image_calib(self.src_Lobe_Per_overlay_arr)
                    self.lobe_per_overlay_extent_1 = self.image_extent(self.src_Lobe_Per_overlay,
                                                                       self.src_Lobe_Per_overlay_arr)
                    self.lobe_per_overlay_extent_2 = self.image_extent(self.src_Lobe_Per_overlay,
                                                                       self.src_Lobe_Per_overlay_arr)
                    self.lobe_per_overlay_extent_ori = self.image_extent(self.src_Lobe_Per_overlay,
                                                                         self.src_Lobe_Per_overlay_arr)

                    self.imv_Lobe_Per_overlay_axial = self.imv_creator(self.lobe_per_overlay_calib[0], levels=(0, 800))
                    self.imv_Lobe_Per_overlay_coronal = self.imv_creator(
                        cv2.rotate(self.lobe_per_overlay_extent_ori[:, 0, :, :], self.calib_code), levels=(0, 800))
                    self.imv_Lobe_Per_overlay_sagittal = self.imv_creator(
                        cv2.rotate(self.lobe_per_overlay_extent_ori[:, :, 0, :], self.calib_code), levels=(0, 800))

                    """ Create scrollbar and label """
                    self.scroll_lobe_per_axial.setSliderPosition(self.lobe_per_overlay_calib.shape[0] // 2)
                    self.scroll_lobe_per_axial.setRange(0, self.src_Lobe_Per_overlay_arr.shape[0] - 1)
                    self.scroll_lobe_per_coronal.setSliderPosition(self.lobe_per_overlay_calib.shape[0] // 2)
                    self.scroll_lobe_per_coronal.setRange(0, self.src_Lobe_Per_overlay_arr.shape[1] - 1)
                    self.scroll_lobe_per_sagittal.setSliderPosition(self.lobe_per_overlay_calib.shape[0] // 2)
                    self.scroll_lobe_per_sagittal.setRange(0, self.src_Lobe_Per_overlay_arr.shape[2] - 1)

                    self.label_lobe_per_axial.setStyleSheet('background-color: black; color: white')
                    self.label_lobe_per_axial.setText(' Axial view\nAxial slice: 90')
                    self.label_lobe_per_coronal.setStyleSheet('background-color: black; color: white')
                    self.label_lobe_per_coronal.setText(' Coronal view\nCoronal slice: 90')
                    self.label_lobe_per_sagittal.setStyleSheet('background-color: black; color: white')
                    self.label_lobe_per_sagittal.setText(' Sagittal view\nSagittal slice: 90 ')

                    """ Arrange Layout """
                    self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.imv_Lobe_Per_overlay_axial, 0, 0,
                                                                               16, 16)
                    self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.scroll_lobe_per_axial, 0, 16, 17, 1)
                    self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.label_lobe_per_axial, 16, 0, 1, 16)

                    self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.imv_Lobe_Per_overlay_coronal, 0, 17,
                                                                               7, 7)
                    self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.label_lobe_per_coronal, 7, 17, 1, 7)
                    self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.scroll_lobe_per_coronal, 0, 25, 8,
                                                                               1)

                    self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.imv_Lobe_Per_overlay_sagittal, 8,
                                                                               17, 8, 7)
                    self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.label_lobe_per_sagittal, 16, 17, 1,
                                                                               7)
                    self.view_Window.lobe_per_layout.LOBE_PER_Layout.addWidget(self.scroll_lobe_per_sagittal, 8, 25, 9,
                                                                               1)
                if self.lung_ven_flag == 1:
                    self.ven_lobe1, self.ven_lobe2, self.ven_lobe3, self.ven_lobe4, self.ven_lobe5 = cal_volume_Per_Ven.cal_percent_Per_Ven(
                        self.src_lung_label, self.Ven_image)
                    self.view_Window.ventilation_show.setText(
                        'VETILATION :\n-ULL : {} %\n-LLL : {} %\n-URL : {} %\n-MRL : {} %\n-LRL : {} %'.format(
                            self.ven_lobe1, self.ven_lobe2, self.ven_lobe3, self.ven_lobe4, self.ven_lobe5))
                    # cap nhat lai LOBE-VENTILATION-OVERLAY
                    self.src_Lobe_Ven_overlay = sitk.LabelOverlay(image=self.src_Ven,
                                                                  labelImage=self.src_lung_label_ero,
                                                                  opacity=0.1, backgroundValue=0.7,
                                                                  colormap=self.red + self.green + self.blue + self.yellow + self.glaucous)
                    self.src_Lobe_Ven_overlay_arr = sitk.GetArrayFromImage(self.src_Lobe_Ven_overlay)

                    self.lobe_ven_overlay_calib = self.image_calib(self.src_Lobe_Ven_overlay_arr)
                    self.lobe_ven_overlay_extent_1 = self.image_extent(self.src_Lobe_Ven_overlay,
                                                                       self.src_Lobe_Ven_overlay_arr)
                    self.lobe_ven_overlay_extent_2 = self.image_extent(self.src_Lobe_Ven_overlay,
                                                                       self.src_Lobe_Ven_overlay_arr)
                    self.lobe_ven_overlay_extent_ori = self.image_extent(self.src_Lobe_Ven_overlay,
                                                                         self.src_Lobe_Ven_overlay_arr)

                    self.imv_Lobe_Ven_overlay_axial = self.imv_creator(self.lobe_ven_overlay_calib[0], levels=(0, 800))
                    self.imv_Lobe_Ven_overlay_coronal = self.imv_creator(
                        cv2.rotate(self.lobe_ven_overlay_extent_ori[:, 0, :, :], self.calib_code), levels=(0, 800))
                    self.imv_Lobe_Ven_overlay_sagittal = self.imv_creator(
                        cv2.rotate(self.lobe_ven_overlay_extent_ori[:, :, 0, :], self.calib_code), levels=(0, 800))

                    """ Create scrollbar and label """
                    self.scroll_lobe_ven_axial.setSliderPosition(self.lobe_ven_overlay_calib.shape[0] // 2)
                    self.scroll_lobe_ven_axial.setRange(0, self.src_Lobe_Ven_overlay_arr.shape[0] - 1)
                    self.scroll_lobe_ven_coronal.setSliderPosition(self.lobe_ven_overlay_calib.shape[0] // 2)
                    self.scroll_lobe_ven_coronal.setRange(0, self.src_Lobe_Ven_overlay_arr.shape[1] - 1)
                    self.scroll_lobe_ven_sagittal.setSliderPosition(self.lobe_ven_overlay_calib.shape[0] // 2)
                    self.scroll_lobe_ven_sagittal.setRange(0, self.src_Lobe_Ven_overlay_arr.shape[2] - 1)

                    self.label_lobe_ven_axial.setStyleSheet('background-color: black; color: white')
                    self.label_lobe_ven_axial.setText(' Axial view\nAxial slice: 90')
                    self.label_lobe_ven_coronal.setStyleSheet('background-color: black; color: white')
                    self.label_lobe_ven_coronal.setText(' Coronal view\nCoronal slice: 90')
                    self.label_lobe_ven_sagittal.setStyleSheet('background-color: black; color: white')
                    self.label_lobe_ven_sagittal.setText(' Sagittal view\nSagittal slice: 90 ')

                    """ Arrange Layout """
                    self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.imv_Lobe_Ven_overlay_axial, 0, 0,
                                                                               16, 16)
                    self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.scroll_lobe_ven_axial, 0, 16, 17, 1)
                    self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.label_lobe_ven_axial, 16, 0, 1, 16)

                    self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.imv_Lobe_Ven_overlay_coronal, 0, 17,
                                                                               7, 7)
                    self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.label_lobe_ven_coronal, 7, 17, 1, 7)
                    self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.scroll_lobe_ven_coronal, 0, 25, 8,
                                                                               1)

                    self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.imv_Lobe_Ven_overlay_sagittal, 8,
                                                                               17, 8, 7)
                    self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.label_lobe_ven_sagittal, 16, 17, 1,
                                                                               7)
                    self.view_Window.lobe_ven_layout.LOBE_VEN_Layout.addWidget(self.scroll_lobe_ven_sagittal, 8, 25, 9,
                                                                               1)

            #  if the 'exit' key is pressed, break from the loop
            elif key == ord("e"):
                break

        cv2.destroyAllWindows()

    def replanning(self):
        # luu file
        file_name = QFileDialog.getSaveFileName(
            parent=self.view_Window,
            caption='SaveFile',

            directory=os.getcwd(),
            filter='Image files (*.nii.gz *.nii)'

        )

        print(file_name)
        sitk.WriteImage(self.src_lung_label, fileName=file_name[0] + '.nii.gz', useCompression=True)
        #
        rm = self.view_Window.virtualPlanning_layout.Planning_Layout.itemAt(0)
        if rm != None:
            rm.widget().deleteLater()

        # Create and compute new render
        spacing = np.round_(self.src.GetSpacing()[2]).astype(int)

        self.full_pred_extent = np.zeros(
            (spacing * self.lung_pred.shape[0], self.lung_pred.shape[1], self.lung_pred.shape[2]))
        self.SP_image_extent = np.zeros(
            (spacing * self.lung_pred.shape[0], self.lung_pred.shape[1], self.lung_pred.shape[2]))

        self.SP_image_thresh = np.zeros(self.SP_image.shape)
        for i in range(self.SP_image.shape[0]):
            ret, self.SP_image_thresh[i] = cv2.threshold(np.int16(self.SP_image[i]), 120, 1, cv2.THRESH_BINARY)

        for i in range(self.lung_pred.shape[0]):
            for j in range(spacing):
                if self.calib_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
                    self.full_pred_extent[spacing * i + j] = self.lung_pred[i]
                    self.SP_image_extent[spacing * i + j] = self.SP_image_thresh[i]
                else:
                    self.full_pred_extent[spacing * i + j] = cv2.flip(
                        cv2.rotate(self.lung_pred[i], cv2.ROTATE_90_CLOCKWISE), 1)
                    self.SP_image_extent[spacing * i + j] = cv2.flip(
                        cv2.rotate(self.SP_image_thresh[i], cv2.ROTATE_90_CLOCKWISE), 1)

        self.VTK_Renderer(lung_pred=self.full_pred_extent, img_data=self.SP_image_extent)

    # def save_img(self):
    #     # save = QFileDialog.getExistingDirectory(caption='Chọn file bạn muốn lưu')
    #     # sitk.WriteImage(self.src_lung_label, fileName= os.path.join(save + '/' +self.a.split('.')[0] + ".roi.nii.gz")  , useCompression=True)
    #     # self.view_Window.save_btn.setEnabled(True)

    #     file_name = QFileDialog.getSaveFileName(
    #         parent=self.view_Window,
    #         caption='SaveFile',

    #         directory=os.getcwd(),
    #         filter='Image files (*.nii.gz *.nii)'

    #     )

    #     print(file_name)
    #     sitk.WriteImage(self.src_lung_label, fileName= file_name[0]+'.nii.gz' , useCompression=True)
    #     #

    def bgr2rgb(self, bgr_image):
        for i in range(bgr_image.shape[0]):
            bgr_image[i] = cv2.cvtColor(bgr_image[i], cv2.COLOR_BGR2RGB)

        return bgr_image

    def color_bgr2rgb_opacity(self, color):
        new_color = [color[2], color[1], color[0], color[3]]
        return new_color

    def color_bgr2rgb(self, color):
        new_color = [color[2], color[1], color[0]]
        return new_color

    

    def image_calib(self, image_src):
        # RGB image
        if len(image_src.shape) == 4:
            self.value = int(np.min(image_src))
            self.color = (255, 255, 255)

            img = np.ones((image_src.shape[0], image_src.shape[1], image_src.shape[2], 3))
        # Grayscale image
        else:
            self.value = int(np.min(image_src))
            self.color = int(np.max(image_src))

            img = np.ones((image_src.shape))

        """ Calib Image Viewer """
        for i in range(image_src.shape[0]):
            # img_buffer[i] = cv2.copyMakeBorder(image_src[i], self.top, self.bottom, self.left, self.right, cv2.BORDER_CONSTANT, value=self.value)
            img[i] = (cv2.flip(cv2.rotate(image_src[i], self.calib_code), 1))

        return img

    def image_extent(self, image_src, image):
        """
            image_src: sitk image format
            image: numpy array format
        """
        spacing = np.round_(image_src.GetSpacing()[2]).astype(int)
        image_extent = np.zeros((spacing * image.shape[0], image.shape[1], image.shape[2], 3))

        for i in range(image.shape[0]):
            for j in range(spacing):
                image_extent[spacing * i + j] = image[i]

        return image_extent

    # tao thanh keo chinh do sang vơi level
    def imv_creator(self, image_src, autoRange=True, levels=(-16, 256)):
        imv = ImageView()
        imv.setImage(image_src, autoRange=autoRange, levels=levels)
        # imv.ui.histogram.hide()
        imv.ui.menuBtn.hide()
        imv.ui.roiBtn.hide()

        return imv

    def VTK_Renderer(self, lung_pred, img_data):
        self.vtkWidget = QVTKRenderWindowInteractor()
        self.view_Window.virtualPlanning_layout.Planning_Layout.addWidget(self.vtkWidget, 0, 0)

        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        """ -------------------------------------------- Load 3D CT image data --------------------------------------------- """
        self.img_data = img_data
        self.max_pixel = int(np.max(self.img_data))
        self.img_data_shape = self.img_data.shape
        # Apply mask into original image
        self.pred = lung_pred
        self.img_data = np.where(self.pred == 1, self.max_pixel + 1, self.img_data)
        self.img_data = np.where(self.pred == 2, self.max_pixel + 2, self.img_data)
        self.img_data = np.where(self.pred == 3, self.max_pixel + 3, self.img_data)
        self.img_data = np.where(self.pred == 4, self.max_pixel + 4, self.img_data)
        self.img_data = np.where(self.pred == 5, self.max_pixel + 5, self.img_data)

        self.img_data = self.img_data.view(type=np.ndarray)
        self.img_data = self.img_data.astype(np.double)
        self.img_data = self.img_data.view(np.double)
        # Import masked image into VTK format vtkImageData
        data_string = img_data.tobytes()
        self.dataImporter = vtk.vtkImageImport()
        self.dataImporter.SetDataScalarTypeToInt()
        self.dataImporter.SetNumberOfScalarComponents(1)
        self.dataImporter.SetDataScalarTypeToDouble()

        self.img_data = np.ascontiguousarray(self.img_data, np.double)
        self.img_data = self.img_data.tobytes(order='F')

        self.dataImporter.SetImportVoidPointer(self.img_data)

        # extent = dataImporter.GetDataExtent()
        self.dataImporter.SetDataExtent(0, self.img_data_shape[0] - 1,
                                        0, self.img_data_shape[1] - 1,
                                        0, self.img_data_shape[2] - 1)
        self.dataImporter.SetWholeExtent(0, self.img_data_shape[0] - 1,
                                         0, self.img_data_shape[1] - 1,
                                         0, self.img_data_shape[2] - 1)

        # self.dataImporter.SetDataSpacing( 1.0, 1.0, 1.0)
        self.dataImporter.SetDataOrigin(0, 0, 0)
        self.dataImporter.Update()
        vtk_image_data = self.dataImporter.GetOutput()

        """ -------------------------------------------- Render 3D CT volume ---------------------------------------------- """
        colors = vtk.vtkNamedColors()

        # Set opacity for each pixel
        alphaChannelFunc = vtk.vtkPiecewiseFunction()
        alphaChannelFunc.AddPoint(0, 0)
        alphaChannelFunc.AddPoint(1, 0.1)
        alphaChannelFunc.AddPoint(self.max_pixel + 1, 0.6)
        alphaChannelFunc.AddPoint(self.max_pixel + 2, 0.6)
        alphaChannelFunc.AddPoint(self.max_pixel + 3, 0.6)
        alphaChannelFunc.AddPoint(self.max_pixel + 4, 0.6)
        alphaChannelFunc.AddPoint(self.max_pixel + 5, 0.6)

        # Set RGB mapping for each pixel
        colorFunc = vtk.vtkColorTransferFunction()
        colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)
        colorFunc.AddRGBPoint(1, 0.38, 0.42, 0.45)
        colorFunc.AddRGBPoint(self.max_pixel + 1, self.red[0], self.red[1], self.red[2])
        colorFunc.AddRGBPoint(self.max_pixel + 2, self.green[0], self.green[1], self.green[2])
        colorFunc.AddRGBPoint(self.max_pixel + 3, self.blue[0], self.blue[1], self.blue[2])
        colorFunc.AddRGBPoint(self.max_pixel + 4, self.yellow[0], self.yellow[1], self.yellow[2])
        colorFunc.AddRGBPoint(self.max_pixel + 5, self.glaucous[0], self.glaucous[1], self.glaucous[2])

        # Apply property to VTK Renderer
        self.volumeProperty = vtk.vtkVolumeProperty()
        self.volumeProperty.SetColor(colorFunc)
        self.volumeProperty.SetScalarOpacity(alphaChannelFunc)

        # Set mapper for 3D CT volume
        self.volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
        self.volumeMapper.SetInputConnection(self.dataImporter.GetOutputPort())

        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.volumeMapper)
        self.volume.SetProperty(self.volumeProperty)

        self.ren.AddActor(self.volume)

        self.iren.Initialize()
        self.iren.Start()


class model_Window:
    def __init__(self):
        super().__init__()

    # Function for automatic 3D region growing
    def lungSegment(self, src_ct_image):
        # None

        # Uncomment to use fast-but-medium-accuarcy mode
        model = mask.get_model('unet', 'LTRCLobes')
        lung_pred = mask.apply(src_ct_image,model)

        # # Uncomment to use slow-but-accurate mode
        # lung_pred = mask.apply_fused(src_ct_image)

        src_lung_label = sitk.GetImageFromArray(lung_pred)
        src_lung_label.SetOrigin(src_ct_image.GetOrigin())
        src_lung_label.SetSpacing(src_ct_image.GetSpacing())
        src_lung_label.SetDirection(src_ct_image.GetDirection())

        return lung_pred, src_lung_label


''' ----------------------------------------------------- Break --------------------------------------------------'''


class ImageView(pg.ImageView):
    def __init__(self, *args, **kwargs):
        super(ImageView, self).__init__(*args, **kwargs)


class Init_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.Init_Layout = QGridLayout(self)

        self.phase3 = QLabel()
        self.phase3.setFrameShape(QFrame.Panel)
        self.phase3.setLineWidth(1)
        self.phase3.setScaledContents(True)
        self.phase3.setObjectName('SPECT/CT')
        self.Init_Layout.addWidget(self.phase3, 0, 0)

        self.setLayout(self.Init_Layout)


class Show_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.Show_Layout = QGridLayout(self)
        self.setLayout(self.Show_Layout)


class LOBE_PER_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.LOBE_PER_Layout = QGridLayout(self)
        self.setLayout(self.LOBE_PER_Layout)


class LOBE_VEN_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.LOBE_VEN_Layout = QGridLayout(self)
        self.setLayout(self.LOBE_VEN_Layout)


class lungSegment_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.Segment_Layout = QGridLayout(self)
        self.setLayout(self.Segment_Layout)


class virtualPlanning_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.Planning_Layout = QGridLayout(self)
        self.setLayout(self.Planning_Layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    view_Window = view_Window()
    view_Window.show()
    model_Window = model_Window()
    control_Window = control_Window(view_Window=view_Window, model_Window=model_Window)
    sys.exit(app.exec_())


