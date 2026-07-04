// tproc_cxl_driver.c
// T-Processor 作为 CXL Type-3 Device 的 Linux 内核驱动骨架
// 对应章锋论文附录B
// MuJoCo-Bench-IDO 硬件参考

#include <linux/module.h>
#include <linux/pci.h>
#include <linux/cdev.h>
#include <linux/interrupt.h>
#include <linux/uaccess.h>
#include <linux/mm.h>
#include <linux/delay.h>

#define TPROC_VENDOR_ID 0x1DE5
#define TPROC_DEVICE_ID 0x0001
#define TPROC_MMIO_BAR  0
#define TPROC_CXL_BAR   2

// T-Proc Register Map (MMIO)
#define TPROC_REG_MAGIC          0x0000
#define TPROC_REG_VERSION        0x0004
#define TPROC_REG_CTRL           0x0008
#define TPROC_REG_STATUS         0x000C
#define TPROC_REG_Q0             0x0100
#define TPROC_REG_ALU_OP         0x0200
#define TPROC_REG_PSI_ETA        0x0300
#define TPROC_REG_KSNAP_HEAD    0x0400
#define TPROC_REG_KSNAP_TAIL    0x0404
#define TPROC_REG_KSNAP_BUF     0x0800

// Control Bits
#define CTRL_RESET              (1 << 0)
#define CTRL_ENABLE             (1 << 1)
#define CTRL_RUN_PHI            (1 << 2)
#define CTRL_IRQ_MASK_KSNAP     (1 << 8)

struct tproc_dev {
    struct pci_dev *pdev;
    void __iomem *mmio_base;
    void __iomem *cxl_base;
    struct cdev cdev;
    dev_t devno;
    void *ksnap_buf_dma;
    dma_addr_t ksnap_dma_handle;
};

static irqreturn_t tproc_irq_handler(int irq, void *data) {
    struct tproc_dev *tproc = data;
    u32 status = readl(tproc->mmio_base + TPROC_REG_STATUS);
    if (status & (1 << 8)) {
        writel(status & ~(1 << 8), tproc->mmio_base + TPROC_REG_STATUS);
        return IRQ_HANDLED;
    }
    return IRQ_NONE;
}

static long tproc_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
    struct tproc_dev *tproc = filp->private_data;
    switch (cmd) {
        case 0x01: // INIT
            writel(CTRL_RESET, tproc->mmio_base + TPROC_REG_CTRL);
            udelay(10);
            writel(CTRL_ENABLE, tproc->mmio_base + TPROC_REG_CTRL);
            break;
        case 0x02: // LOAD_EML
            if (copy_from_user(tproc->cxl_base, (void __user *)arg, 4*1024*1024))
                return -EFAULT;
            break;
        case 0x03: // TRIGGER_PHI
            writel(CTRL_RUN_PHI, tproc->mmio_base + TPROC_REG_CTRL);
            while (readl(tproc->mmio_base + TPROC_REG_STATUS) & (1 << 0));
            break;
        case 0x04: // READ_KSNAP
            break;
    }
    return 0;
}

static int tproc_mmap(struct file *filp, struct vm_area_struct *vma) {
    struct tproc_dev *tproc = filp->private_data;
    unsigned long phys = virt_to_phys(tproc->ksnap_buf_dma);
    if (remap_pfn_range(vma, vma->vm_start, phys >> PAGE_SHIFT,
                       vma->vm_end - vma->vm_start, vma->vm_page_prot))
        return -EAGAIN;
    return 0;
}

static int tproc_probe(struct pci_dev *pdev, const struct pci_device_id *id) {
    struct tproc_dev *tproc;
    int ret;

    tproc = devm_kzalloc(&pdev->dev, sizeof(*tproc), GFP_KERNEL);
    pci_set_drvdata(pdev, tproc);
    tproc->pdev = pdev;

    ret = pci_enable_device(pdev);
    ret = pci_request_regions(pdev, "tproc");
    tproc->mmio_base = pci_iomap(pdev, TPROC_MMIO_BAR, 0);
    tproc->cxl_base = pci_iomap(pdev, TPROC_CXL_BAR, 0);
    tproc->ksnap_buf_dma = dma_alloc_coherent(&pdev->dev, 4*1024*1024,
                                              &tproc->ksnap_dma_handle, GFP_KERNEL);
    ret = request_irq(pdev->irq, tproc_irq_handler, IRQF_SHARED, "tproc", tproc);
    return 0;
}

static struct pci_driver tproc_driver = {
    .name = "tproc",
    .id_table = (struct pci_device_id[]){{ TPROC_VENDOR_ID, TPROC_DEVICE_ID, PCI_ANY_ID, PCI_ANY_ID }, {0}},
    .probe = tproc_probe,
};

module_pci_driver(tproc_driver);
MODULE_LICENSE("GPL");

// IOCTL命令:
//   0x01 INIT: 复位并使能T-Processor
//   0x02 LOAD_EML: 通过CXL加载4MB EML节点
//   0x03 TRIGGER_PHI: 触发Φ运算（八元数流贯演化）
//   0x04 READ_KSNAP: 读取κ-Snap审计日志
//
// 寄存器映射:
//   0x0000 MAGIC: 魔数标识
//   0x0004 VERSION: 版本号
//   0x0008 CTRL: 控制寄存器 (RESET/ENABLE/RUN_PHI/IRQ_MASK)
//   0x000C STATUS: 状态寄存器
//   0x0100 Q0: 八元数q寄存器
//   0x0200 ALU_OP: ALU操作码
//   0x0300 PSI_ETA: Ψ-锚与η残差
//   0x0400 KSNAP_HEAD: κ-Snap环形缓冲区头指针
//   0x0404 KSNAP_TAIL: κ-Snap环形缓冲区尾指针
//   0x0800 KSNAP_BUF: κ-Snap缓冲区起始地址
