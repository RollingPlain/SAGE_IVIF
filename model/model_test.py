import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, dim: int = 32): 
        super(Network, self).__init__()


        self.module1 = nn.Sequential(
            nn.Conv2d(2, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.01)
        )


        self.downsample1 = nn.MaxPool2d(2)


        self.module2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),  
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.01)
        )


        self.downsample2 = nn.MaxPool2d(2)


        self.module3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1), 
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.01)
        )


        self.adjust_channels4 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.module4 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),  
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.adjust_channels5 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.01)
        )


        self.module5 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1), 
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.final_decoder = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
    def forward(self, vi, ir):

        src = torch.cat([vi, ir], dim=1)  

        x1 = self.module1(src)            

        x2 = self.downsample1(x1)         
        x2 = self.module2(x2)             


        x3 = self.downsample2(x2)         
        x3 = self.module3(x3)             

        x4 = F.interpolate(x3, size=(x2.size(2), x2.size(3)), mode='bilinear', align_corners=False) 
        x4 = torch.cat([x4, x2], dim=1)   
        x4 = self.adjust_channels4(x4)    
        x4 = self.module4(x4)             

        x5 = F.interpolate(x4, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=False)          

        x5 = torch.cat([x5, x1], dim=1)   
        x5 = self.adjust_channels5(x5)    
        x5 = self.module5(x5)             


        out = self.final_decoder(x5)      
        intermediate_outputs = (x1, x2, x3, x4, x5)
        return out, intermediate_outputs
