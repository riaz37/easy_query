"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";
import { CompanyInfoStepProps } from "../types";

export function CompanyInfoStep({
  companyName,
  setCompanyName,
  description,
  setDescription,
  address,
  setAddress,
  contactEmail,
  setContactEmail,
  setCurrentStep,
  onClose,
}: CompanyInfoStepProps) {
  const handleNext = () => {
    if (!companyName.trim()) {
      toast.error("Company name is required");
      return;
    }
    setCurrentStep("database-config");
  };

  const handlePrevious = () => {
    // This is the first step, so we close the modal
    onClose();
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Company Name */}
        <div className="space-y-2">
          <Label htmlFor="companyName" className="modal-label-enhanced">
            Company Name <span className="text-red-500">*</span>
          </Label>
          <Input
            id="companyName"
            value={companyName}
            onChange={(e) => setCompanyName(e.target.value)}
            placeholder="Acme Inc."
            className="modal-input-enhanced h-11"
            required
          />
        </div>

        {/* Contact Email */}
        <div className="space-y-2">
          <Label htmlFor="contactEmail" className="modal-label-enhanced">
            Contact Email
          </Label>
          <Input
            id="contactEmail"
            type="email"
            value={contactEmail}
            onChange={(e) => setContactEmail(e.target.value)}
            placeholder="contact@company.com"
            className="modal-input-enhanced h-11"
          />
        </div>

        {/* Address - Full Width */}
        <div className="md:col-span-2 space-y-2">
          <Label htmlFor="address" className="modal-label-enhanced">
            Company Address
          </Label>
          <Input
            id="address"
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            placeholder="123 Business St, City, Country"
            className="modal-input-enhanced h-11"
          />
        </div>

        {/* Description - Full Width */}
        <div className="md:col-span-2 space-y-2">
          <div className="flex justify-between items-center">
            <Label htmlFor="description" className="modal-label-enhanced">
              About the Company
            </Label>
            <span className="text-xs text-gray-500">
              {description.length}/500
            </span>
          </div>
          <Textarea
            id="description"
            value={description}
            onChange={(e) => setDescription(e.target.value.slice(0, 500))}
            placeholder="Tell us about your company's mission, values, and what makes it unique..."
            className="modal-textarea-enhanced min-h-[100px] resize-none"
          />
        </div>
      </div>

      {/* Navigation */}
      <div className="modal-footer-enhanced">
        <Button
          variant="outline"
          onClick={handlePrevious}
          className="modal-button-secondary w-full sm:w-auto"
        >
          Back
        </Button>
        <Button onClick={handleNext} className="modal-button-primary w-full sm:w-auto">
          Next
        </Button>
      </div>
    </div>
  );
}
