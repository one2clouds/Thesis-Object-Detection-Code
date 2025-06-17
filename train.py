import torchvision 
import torch.nn as nn 
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.ops as ops
from data import get_data
import torch 
from torchvision.ops.giou_loss import generalized_box_iou_loss
from tqdm import tqdm
import wandb 
from torchvision.ops import box_iou
import numpy as np 


def create_model(num_classes=6):
    model_backbone = torchvision.models.resnet101(weights="DEFAULT")

    conv1 = model_backbone.conv1
    bn1 = model_backbone.bn1
    relu = model_backbone.relu
    max_pool = model_backbone.maxpool
    layer1 = model_backbone.layer1
    layer2 = model_backbone.layer2
    layer3 = model_backbone.layer3
    layer4 = model_backbone.layer4

    backbone = nn.Sequential(conv1, bn1, relu, max_pool, layer1, layer2, layer3, layer4)
    backbone.out_channels = 2048

    # Here, we are using 5x3 anchors.
    # Meaning, anchors with 5 different sizes and 3 different aspect ratios.
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = ops.MultiScaleRoIAlign(
        featmap_names=["0"], output_size=7, sampling_ratio=2
    ) # variable size region into fixed size output. Into 7x7 grid 

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
    )
    return model


def calculate_metrics(predictions, targets, iou_threshold=0.5, num_classes=6):
    """
    Calculate precision, recall, AP for object detection
    """
    all_precisions = []
    all_recalls = []
    all_aps = []
    all_f1_scores = []
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        pred_labels = pred['labels']
        
        true_boxes = target['boxes']
        true_labels = target['labels']
        
        class_precisions = []
        class_recalls = []
        class_aps = []
        class_f1s = []
        
        for class_id in range(num_classes):
            # Filter predictions and targets for current class
            pred_mask = pred_labels == class_id
            true_mask = true_labels == class_id
            
            pred_boxes_class = pred_boxes[pred_mask]
            pred_scores_class = pred_scores[pred_mask]
            true_boxes_class = true_boxes[true_mask]
            
            if len(true_boxes_class) == 0 and len(pred_boxes_class) == 0:
                # No ground truth and no predictions for this class
                class_precisions.append(1.0)
                class_recalls.append(1.0)
                class_aps.append(1.0)
                class_f1s.append(1.0)
                continue
            elif len(true_boxes_class) == 0:
                # No ground truth but have predictions (all false positives)
                class_precisions.append(0.0)
                class_recalls.append(0.0)
                class_aps.append(0.0)
                class_f1s.append(0.0)
                continue
            elif len(pred_boxes_class) == 0:
                # Have ground truth but no predictions (all false negatives)
                class_precisions.append(0.0)
                class_recalls.append(0.0)
                class_aps.append(0.0)
                class_f1s.append(0.0)
                continue
            
            # Calculate IoU between all predicted and true boxes for this class
            if len(pred_boxes_class) > 0 and len(true_boxes_class) > 0:
                ious = box_iou(pred_boxes_class, true_boxes_class)
                
                # Sort predictions by confidence score (descending)
                sorted_indices = torch.argsort(pred_scores_class, descending=True)
                
                tp = 0
                fp = 0
                matched_gt = set()
                
                precisions = []
                recalls = []
                
                for idx in sorted_indices:
                    best_iou, best_gt_idx = torch.max(ious[idx], dim=0)
                    
                    if best_iou >= iou_threshold and best_gt_idx.item() not in matched_gt:
                        tp += 1
                        matched_gt.add(best_gt_idx.item())
                    else:
                        fp += 1
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / len(true_boxes_class) if len(true_boxes_class) > 0 else 0
                    
                    precisions.append(precision)
                    recalls.append(recall)
                
                if len(precisions) > 0:
                    # Calculate AP using precision-recall curve
                    precisions = np.array(precisions)
                    recalls = np.array(recalls)
                    
                    # Add points at recall 0 and 1
                    precisions = np.concatenate(([0], precisions, [0]))
                    recalls = np.concatenate(([0], recalls, [1]))
                    
                    # Compute AP using trapezoidal rule
                    ap = np.trapezoid(precisions, recalls)
                    
                    final_precision = precisions[-2] if len(precisions) > 1 else 0
                    final_recall = recalls[-2] if len(recalls) > 1 else 0
                    f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0
                    
                    class_precisions.append(final_precision)
                    class_recalls.append(final_recall)
                    class_aps.append(ap)
                    class_f1s.append(f1)
                else:
                    class_precisions.append(0.0)
                    class_recalls.append(0.0)
                    class_aps.append(0.0)
                    class_f1s.append(0.0)
            else:
                class_precisions.append(0.0)
                class_recalls.append(0.0)
                class_aps.append(0.0)
                class_f1s.append(0.0)
        
        all_precisions.append(np.mean(class_precisions))
        all_recalls.append(np.mean(class_recalls))
        all_aps.append(np.mean(class_aps))
        all_f1_scores.append(np.mean(class_f1s))
    
    return {
        'precision': np.mean(all_precisions),
        'recall': np.mean(all_recalls),
        'mAP': np.mean(all_aps),
        'f1_score': np.mean(all_f1_scores)
    }


def calculate_map_at_different_ious(predictions, targets, num_classes=7):
    """Calculate mAP at different IoU thresholds (0.5:0.05:0.95)"""
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    for iou_thresh in iou_thresholds:
        metrics = calculate_metrics(predictions, targets, iou_threshold=iou_thresh, num_classes=num_classes)
        aps.append(metrics['mAP'])
    return np.mean(aps) 


if __name__ == "__main__":

    wandb.init(
        project="Steel Scrap Detection", 
        config={
            "epochs":2000, 
            "batch_size": 8,
            "lr":0.005, 
        }
    )

    TRAIN_IMAGES = "/home/shirshak/Thesis_Data/Steel_Scrap_detection/train/images"
    TRAIN_LABELS = "/home/shirshak/Thesis_Data/Steel_Scrap_detection/train/labels"
    VAL_IMAGES = "/home/shirshak/Thesis_Data/Steel_Scrap_detection/valid/images"
    VAL_LABELS = "/home/shirshak/Thesis_Data/Steel_Scrap_detection/valid/labels"

    classes = ['Capacitor', 'Cylinder', 'Motor', 'Shock absorber', 'container']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    _, _, train_loader, val_loader = get_data(TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS, wandb.config.batch_size)
    

    model = create_model(num_classes=6)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config.lr, momentum=0.9, weight_decay=0.0005)
    

    best_val_loss = float('inf')
    best_map_50_95 = 0.0

    patience = 500
    patience_counter = 0

    for epoch in range(wandb.config.epochs):
        train_loss = 0.0
        train_box_loss = 0.0
        train_class_loss = 0.0
        num_batches = 0
        train_all_predictions = []
        train_all_targets = []
        model.train()

        for images, targets in tqdm(train_loader):
            images = [img.cuda() for img in images]
            targets = [{k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())            

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            train_loss += losses.item()
            train_box_loss += loss_dict["loss_box_reg"].item()
            train_class_loss += loss_dict["loss_classifier"].item()

            num_batches += 1

        #     # Get Predictions for metrics calculation
        #     model.eval()
        #     predictions = model(images)
        #     train_predictions = []
        #     train_targets = []
        #     for pred in predictions : 
        #         train_predict = {k: v.cpu() for k, v in pred.items()}
        #         train_predictions.append(train_predict)
        #     for targ in targets: 
        #         train_targ = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in targ.items()}
        #         train_targets.append(train_targ)
        #     train_all_predictions.extend(train_predictions)
        #     train_all_targets.extend(train_targets)
        # train_metrics_50 = calculate_metrics(train_all_predictions, train_all_targets, iou_threshold=0.5)
        # # mAP@0.75
        # train_metrics_75 = calculate_metrics(train_all_predictions, train_all_targets, iou_threshold=0.75)
        # # mAP@0.5:0.95
        # train_map_50_95 = calculate_map_at_different_ious(train_all_predictions, train_all_targets)

        avg_train_loss = train_loss / num_batches
        avg_train_box_loss = train_box_loss / num_batches
        avg_train_class_loss = train_class_loss / num_batches

        # wandb.log({
        #     "train/train_loss":avg_train_loss, 
        #     "train/precision_50": train_metrics_50['precision'],
        #     "train/recall_50": train_metrics_50['recall'],
        #     "train/f1_score_50": train_metrics_50['f1_score'],
        #     "train/mAP_50": train_metrics_50['mAP'],
        #     "train/mAP_75": train_metrics_75['mAP'],
        #     "train/mAP_50_95": train_map_50_95,
        #     "epoch": epoch + 1
        #     })
        val_loss = 0.0
        val_batches = 0

        val_box_loss = 0.0 
        val_class_loss = 0.0

        val_all_predictions = []
        val_all_targets = []

        with torch.no_grad():
            for images, targets in tqdm(val_loader):
                images = [img.cuda() for img in images]
                targets = [{k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                
                model.train()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

                val_box_loss += loss_dict["loss_box_reg"].item()
                val_class_loss += loss_dict["loss_classifier"].item()

                val_batches += 1

                model.eval()
                predictions = model(images)
                val_predictions = []
                val_targets = []

                for pred in predictions : 
                    val_predict = {k: v.cpu() for k, v in pred.items()}
                    val_predictions.append(val_predict)

                for targ in targets: 
                    val_targ = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in targ.items()}
                    val_targets.append(val_targ)

                val_all_predictions.extend(val_predictions)
                val_all_targets.extend(val_targets)

            avg_val_loss = val_loss / val_batches
            avg_val_box_loss = val_box_loss / val_batches
            avg_val_class_loss = val_class_loss / val_batches
            
            val_metrics_50 = calculate_metrics(val_all_predictions, val_all_targets, iou_threshold=0.5)
        
            val_metrics_75 = calculate_metrics(val_all_predictions, val_all_targets, iou_threshold=0.75)
            
            val_map_50_95 = calculate_map_at_different_ious(val_all_predictions, val_all_targets)
                    
            print(
                f"\n Epoch {epoch + 1}/{wandb.config.epochs} Summary:\n"
                f"{'-'*90}\n"
                f"🔹Training:\n"
                f"  Training Loss : {avg_train_loss:.4f} |"
                f"  Training Box Loss : {avg_train_box_loss:.4f} |"
                f"  Training Class Loss : {avg_train_class_loss:.4f} | \n"
                f"🔹Validation:\n"
                f"Validation Loss : {avg_val_loss:.4f} | "
                f"Validation Box Loss : {avg_val_box_loss:.4f} |"
                f"Validation Class Loss : {avg_val_class_loss:.4f} |"
                f"Precision@0.5 : {val_metrics_50['precision']:.4f} | "
                f"Recall@0.5 : {val_metrics_50['recall']:.4f} | "
                f"F1-Score@0.5 : {val_metrics_50['f1_score']:.4f} | "
                f"mAP@0.5 : {val_metrics_50['mAP']:.4f} | "
                f"mAP@0.75 : {val_metrics_75['mAP']:.4f} | "
                f"mAP@0.5:0.95 : {val_map_50_95:.4f}\n"
                f"{'-'*90}"
            )

            wandb.log({"train/train_loss":avg_train_loss, 
                    "train/train_box_loss":avg_train_box_loss, 
                    "train/train_class_loss":avg_train_class_loss
                    })
            wandb.log({"epoch": epoch + 1})
            wandb.log({
                "val/val_loss":avg_val_loss, 
                "val/val_box_loss":avg_val_box_loss,
                "val/val_class_loss":avg_val_class_loss,
                "val/precision_50": val_metrics_50['precision'],
                "val/recall_50": val_metrics_50['recall'],
                "val/f1_score_50": val_metrics_50['f1_score'],
                "val/mAP_50": val_metrics_50['mAP'],
                "val/mAP_75": val_metrics_75['mAP'],
                "val/mAP_50_95": val_map_50_95,
                })        


        checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_map_50': val_metrics_50['mAP'],
                'val_map_75': val_metrics_75['mAP'],
                'val_map_50_95': val_map_50_95,
            }
        torch.save(checkpoint, "recent_epoch_model.pth")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_map_50': val_metrics_50['mAP'],
                'val_map_75': val_metrics_75['mAP'],
                'val_map_50_95': val_map_50_95,
            }
            torch.save(checkpoint, "best_model_val_loss.pth")
            print(f"New best model saved at epoch {epoch+1} with val_loss: {avg_val_loss:.4f}")

        if val_map_50_95 > best_map_50_95:
            best_map_50_95 = val_map_50_95
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_map_50': val_metrics_50['mAP'],
                'val_map_75': val_metrics_75['mAP'],
                'val_map_50_95': val_map_50_95,
            }
            torch.save(checkpoint, "best_map50_95.pth")
            print(f"Best mAP@0.5:0.95 model saved at epoch {epoch+1}: {val_map_50_95:.4f}")


        if avg_val_loss > best_val_loss:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_map_50': val_metrics_50['mAP'],
                'val_map_75': val_metrics_75['mAP'],
                'val_map_50_95': val_map_50_95,
            }
            torch.save(checkpoint, "last_epoch_model_ckpt.pth")
            print(f"Model saved")
            exit(0)
    
    wandb.finish()

