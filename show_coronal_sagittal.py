def edit_manaul(self):
    self.SP_update_flag = 1

    option = self.view_Window.segment_btn.currentText()
    if option == '> Lung Segmentation':
        self.overlay = self.bgr2rgb(self.lung_overlay.copy())
        self.label = self.lung_pred.copy()
        cur_slice = self.scroll_lung_axial.value()


    self.image_edit = self.overlay[cur_slice].copy()
    self.clone = self.overlay[cur_slice].copy()
    self.label_clone = self.label[cur_slice].copy()

    self.cur_annotation = []
    self.refPt = []
    self.count = 0

    self.draw_save_flag = 0
    self.seed = []

    def draw_line(event, x, y, flags, param):
        # left click for drawing the next line
        if event == cv2.EVENT_LBUTTONDOWN and self.draw_save_flag == 0:
            self.refPt.append((x, y))
            cv2.line(self.image_edit, self.refPt[self.count - 1], self.refPt[self.count], (0, 255, 0), 1)
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

            self.image_edit = self.clone.copy()
            for i in range(len(self.cur_annotation)):
                refPt_local = self.cur_annotation[i]
                count1 = 1
                for j in range(len(refPt_local) - 1):
                    cv2.line(self.image_edit, refPt_local[count1 - 1], refPt_local[count1], (0, 255, 0), 1)
                    count1 = count1 + 1

            del self.refPt[self.count - 1]
            self.count = self.count - 1
            for i in range(1, self.count):
                cv2.line(self.image_edit, self.refPt[i - 1], self.refPt[i], (0, 255, 0), 1)

        elif event == cv2.EVENT_LBUTTONDOWN and self.draw_save_flag == 1:
            self.seed.append((x, y))

    cv2.namedWindow("image editor", cv2.WINDOW_NORMAL)
    cv2.namedWindow("edit result", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image editor", draw_line)

    while True:
        # display the image and wait for a keypress
        cv2.imshow("image editor", self.image_edit)
        key = cv2.waitKey(1) & 0xFF
        # if the 'reset' key is pressed, reset all
        if key == ord("r"):
            self.image_edit = self.clone.copy()
            self.label[cur_slice] = self.label_clone.copy()
            self.cur_annotation = []
            self.refPt = []
            self.count = 0
            self.draw_save_flag = 0
            self.seed = []

        # mark the end of the current contour
        elif key == ord(" "):
            if len(self.refPt) != 0:
                # add current contour to current annotation
                self.cur_annotation.append(self.refPt)
                # reset refPt and count
                self.refPt = []
                self.count = 0

        # next slice
        elif key == ord("w") and cur_slice < self.overlay.shape[0] - 1:
            # Update slice
            cur_slice = cur_slice + 1
            # Update current annotation, clone(cache), image_edit
            self.image_edit = self.overlay[cur_slice].copy()
            self.clone = self.overlay[cur_slice].copy()
            self.label_clone = self.label[cur_slice].copy()

            self.cur_annotation = []
            self.refPt = []
            self.count = 0


        # back slice
        elif key == ord("s") and cur_slice != 0:
            # Update slice
            cur_slice = cur_slice - 1
            # Update current annotation, clone(cache), image_edit
            self.image_edit = self.overlay[cur_slice].copy()
            self.clone = self.overlay[cur_slice].copy()
            self.label_clone = self.label[cur_slice].copy()

            self.cur_annotation = []
            self.refPt = []
            self.count = 0


        # Press 'a' to confirm drawing and select seed point
        elif key == ord("a"):
            self.draw_save_flag = 1

        # Display edit result of the current slide
        elif key == ord("d") and self.draw_save_flag == 1 and len(self.seed) != 0:
            for i in range(len(self.cur_annotation)):
                self.refPt = self.cur_annotation[i]
                self.count = 0
                for j in range(len(self.refPt) - 1):
                    cv2.line(self.label[cur_slice], self.refPt[self.count], self.refPt[self.count + 1], 0, 2)
                    self.count = self.count + 1

            numLabels, labels, stats, __ = cv2.connectedComponentsWithStats(self.label[cur_slice])

            # print(self.seed, numLabels)
            for i in range(len(self.seed)):
                x = self.seed[i][0]
                y = self.seed[i][1]
                labels[labels == labels[y, x]] = 0

            labels[labels != 0] = 1
            self.label[cur_slice] = labels


            # Display edited result

            origin = self.src_lung_label.GetOrigin()
            spacing = self.src_lung_label.GetSpacing()
            direction = self.src_lung_label.GetDirection()

            self.lung_pred = self.label
            self.src_lung_label = sitk.GetImageFromArray(self.lung_pred)
            self.src_lung_label.SetOrigin(origin)
            self.src_lung_label.SetSpacing(spacing)
            self.src_lung_label.SetDirection(direction)
            self.src_lung_overlay = sitk.LabelOverlay(image=self.src_norm, labelImage=self.src_lung_label,
                                                      opacity=0.1, backgroundValue=0, colormap=self.green)

            self.lung_overlay = sitk.GetArrayFromImage(self.src_lung_overlay)
            self.overlay = self.bgr2rgb(self.lung_overlay.copy())

            cv2.imshow("edit result", cv2.cvtColor(self.lung_overlay[cur_slice], cv2.COLOR_BGR2RGB))



            self.seed = []
            self.draw_save_flag = 0

        # Confirm and save editted result
        elif key == ord("c"):
            # Display edited result
            origin = self.src_lung_label.GetOrigin()
            spacing = self.src_lung_label.GetSpacing()
            direction = self.src_lung_label.GetDirection()

            self.src_lung_label = sitk.GetImageFromArray(self.lung_pred)
            self.src_lung_label.SetOrigin(origin)
            self.src_lung_label.SetSpacing(spacing)
            self.src_lung_label.SetDirection(direction)
            self.src_lung_overlay = sitk.LabelOverlay(image=self.src_norm, labelImage=self.src_lung_label,
                                                      opacity=0.1, backgroundValue=0, colormap=self.green)

            self.lung_overlay = sitk.GetArrayFromImage(self.src_lung_overlay)

            self.lung_overlay_calib = self.image_calib(self.lung_overlay)
            self.lung_overlay_extent_1 = self.image_extent(self.src_lung_overlay, self.lung_overlay)
            self.lung_overlay_extent_2 = self.image_extent(self.src_lung_overlay, self.lung_overlay)
            self.lung_overlay_extent_ori = self.image_extent(self.src_lung_overlay, self.lung_overlay)

            self.scroll_display(view='axial')
            self.scroll_display(view='coronal')
            self.scroll_display(view='sagittal')



            break

        # if the 'exit' key is pressed, break from the loop
        elif key == ord("e"):
            break

    cv2.destroyAllWindows()
    # self.spectct_viewer()