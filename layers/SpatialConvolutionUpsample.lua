local SpatialConvolutionUpsample, parent = torch.class('nn.SpatialConvolutionUpsample','nn.SpatialConvolutionMM')

function SpatialConvolutionUpsample:__init(nInputPlane, nOutputPlane, kW, kH, factor)
   factor = factor or 2
   assert(kW and kH and nInputPlane and nOutputPlane)
   assert(kW % 2 == 1, 'kW has to be odd')
   assert(kH % 2 == 1, 'kH has to be odd')
   self.factor = 2
   self.kW = kW
   self.kH = kH
   self.nInputPlaneU = nInputPlane
   self.nOutputPlaneU = nOutputPlane
   parent.__init(self, nInputPlane, nOutputPlane * factor * factor, kW, kH, 1, 1, (kW-1)/2)
end

function SpatialConvolutionUpsample:updateOutput(input)
   self.output = parent.updateOutput(self, input)
   if input:dim() == 4 then
      self.h = input:size(3)
      self.w = input:size(4)
   else
      self.h = input:size(2)
      self.w = input:size(3)
   end
   self.output = self.output:view(self.nOutputPlaneU, self.h*self.factor, self.w*self.factor)
   return self.output
end

function SpatialConvolutionUpsample:updateGradInput(input, gradOutput)
   gradOutput = gradOutput:view(self.nOutputPlaneU*self.factor*self.factor, self.h, self.w)
   self.gradInput = parent.updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function SpatialConvolutionUpsample:accGradParameters(input, gradOutput, scale)
   gradOutput = gradOutput:view(self.nOutputPlaneU*self.factor*self.factor, self.h, self.w)
   parent.accGradParameters(self, input, gradOutput)
end
