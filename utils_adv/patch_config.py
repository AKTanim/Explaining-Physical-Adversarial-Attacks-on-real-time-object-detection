from torch import optim


class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        # self.img_dir = "./inria/test_run/sample_1/rough_test/"  # train images location
        # self.lab_dir = "./inria/test_run/sample_1/rough_test/yolo_labels/"  # train labels location
        # self.img_dir = "./inria/train/"  # train YOLO labels location
        # self.lab_dir = "./inria/train/yolo_labels/"  # train labels YOLO location
        self.img_dir = "./images/inria/test/images/"  # test set
        self.lab_dir = "./images/inria/test/labels/"  # test set

        # self.img_dir = "./images/two_images/"  # test set
        # self.lab_dir = "./images/two_images/"  # test set

        # self.cfgfile_yolov2 = "./cfg/yolo.cfg"
        # self.weightfile_yolov2 = "./weights/yolov2.weights"
        # self.ssdmbntv1_model_path = "./weights/mbntv1_ssd_voc.pth"

        # coco and voc labels
        # Download the labels
        self.coco_path = "./images/coco/coco_labels.txt"
        self.voc_path = "./images/voc/voc_labels.txt"
        
        self.printfile = "./utils_adv/non_printability/30values.txt"
        
        self.patch_size = 300
        self.start_learning_rate = 0.03

        self.patch_name = 'base'

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        # reduce learning rate when a metric has stopped learning (keras??)
        # In min mode, lr will be reduced when the quantity monitored has stopped decreasing;
        # in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
        # patience is the no of epochs to monitor

        self.max_tv = 0
        self.batch_size = 8
        self.loss_target = lambda obj, cls: obj*cls #for yolo only


class Experiment1(BaseConfig):
    """
    Model that uses a maximum total variation, tv cannot go below this point.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment1'
        self.max_tv = 0.165


class Experiment2HighRes(Experiment1):
    """
    Higher res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 400
        self.patch_name = 'Exp2HighRes'

class Experiment3LowRes(Experiment1):
    """
    Lower res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 100
        self.patch_name = "Exp3LowRes"

class Experiment4ClassOnly(Experiment1):
    """
    Only minimise class score.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment4ClassOnly'
        self.loss_target = lambda obj, cls: cls




class Experiment1Desktop(Experiment1):
    """
    """

    def __init__(self):
        """
        Change batch size.
        """
        super().__init__()

        self.batch_size = 8
        self.patch_size = 400


class ReproducePaperObj(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.batch_size = 8
        self.patch_size = 300

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj #for yolo only


patch_configs = {
    "base": BaseConfig,
    "exp1": Experiment1,
    "exp1_des": Experiment1Desktop,
    "exp2_high_res": Experiment2HighRes,
    "exp3_low_res": Experiment3LowRes,
    "exp4_class_only": Experiment4ClassOnly,
    "paper_obj": ReproducePaperObj
}
