/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.rknn;

import org.opencv.core.Point;
import org.opencv.core.Rect2d;

public class RknnJNI {
    public static enum ModelVersion {
        YOLO_V5,
        YOLO_V8,
        YOLO_V11
    }

    public static class RknnResult {
        public RknnResult(
            int left, int top, int right, int bottom, float conf, int class_id
        ) {
            this.conf = conf;
            this.class_id = class_id;
            this.rect = new Rect2d(new Point(left, top), new Point(right, bottom));
        }
        
        public final Rect2d rect;
        public final float conf;
        public final int class_id;

        @Override
        public String toString() {
            return "RknnResult [rect=" + rect + ", conf=" + conf + ", class_id=" + class_id + "]";
        }

        @Override
        public int hashCode() {
            final int prime = 31;
            int result = 1;
            result = prime * result + ((rect == null) ? 0 : rect.hashCode());
            result = prime * result + Float.floatToIntBits(conf);
            result = prime * result + class_id;
            return result;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj)
                return true;
            if (obj == null)
                return false;
            if (getClass() != obj.getClass())
                return false;
            RknnResult other = (RknnResult) obj;
            if (rect == null) {
                if (other.rect != null)
                    return false;
            } else if (!rect.equals(other.rect))
                return false;
            if (Float.floatToIntBits(conf) != Float.floatToIntBits(other.conf))
                return false;
            if (class_id != other.class_id)
                return false;
            return true;
        }
    }

    /**
     * Create a RKNN detector. Returns valid pointer on success, or NULL on error
     * @param modelPath Absolute path to the model on disk
     * @param numClasses How many classes. MUST MATCH or native code segfaults
     * @param modelVer Which model is being used. Detections will be incorrect if not set to corrresponding model.
     * @return Pointer to the detector in native memory
     */
    public static native long create(String modelPath, int numClasses, int modelVer, int coreNum);

    /**
     * Given an already running detector, change the bitmask controlling which
     * of the 3 cores the model is running on
     * @param ptr Pointer to detector in native memory
     * @param desiredCore Which of the three cores to operate on
     * @return return code of rknn_set_core_mask call, indicating success or failure
     */
    public static native int setCoreMask(long ptr, int desiredCore);
    
    /**
     * Delete all native resources assocated with a detector
     */
    public static native long destroy(long ptr);

    /**
     * Run detction
     * @param detectorPtr Pointer to detector created above
     * @param imagePtr Pointer to a cv::Mat input image
     * @param nmsThresh 
     * @param boxThresh
     */
    public static native RknnResult[] detect(
        long detectorPtr, long imagePtr, double nmsThresh, double boxThresh
    );
}
