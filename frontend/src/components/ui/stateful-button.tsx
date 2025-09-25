"use client";
import { cn } from "@/lib/utils";
import React from "react";
import { motion, AnimatePresence, useAnimate } from "motion/react";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  className?: string;
  children: React.ReactNode;
  mode?: "light" | "dark";
}

export const StatefulButton = ({
  className,
  children,
  mode = "dark",
  ...props
}: ButtonProps) => {
  const [scope, animate] = useAnimate();

  const animateLoading = async () => {
    try {
      const loaderElement = scope.current?.querySelector(".loader");
      if (loaderElement) {
        await animate(
          ".loader",
          {
            width: "20px",
            scale: 1,
            display: "block",
          },
          {
            duration: 0.2,
          }
        );
      }
    } catch (error) {
      console.warn("Animation error in animateLoading:", error);
    }
  };

  const animateSuccess = async () => {
    try {
      const loaderElement = scope.current?.querySelector(".loader");
      const checkElement = scope.current?.querySelector(".check");

      if (loaderElement) {
        await animate(
          ".loader",
          {
            width: "0px",
            scale: 0,
            display: "none",
          },
          {
            duration: 0.2,
          }
        );
      }

      if (checkElement) {
        await animate(
          ".check",
          {
            width: "20px",
            scale: 1,
            display: "block",
          },
          {
            duration: 0.2,
          }
        );

        await animate(
          ".check",
          {
            width: "0px",
            scale: 0,
            display: "none",
          },
          {
            delay: 2,
            duration: 0.2,
          }
        );
      }
    } catch (error) {
      console.warn("Animation error in animateSuccess:", error);
    }
  };

  const handleClick = async (event: React.MouseEvent<HTMLButtonElement>) => {
    if (props.disabled) return;
    await animateLoading();
    await props.onClick?.(event);
    await animateSuccess();
  };

  const {
    onClick,
    onDrag,
    onDragStart,
    onDragEnd,
    onAnimationStart,
    onAnimationEnd,
    ...buttonProps
  } = props;

  return (
    <motion.button
      layout
      layoutId="button"
      ref={scope}
      style={{
        display: "flex",
        minWidth: "100px",
        minHeight: "38px",
        padding: "0px 24px",
        justifyContent: "center",
        alignItems: "center",
        gap: "12px",
        borderRadius: "32px",
        border: props.disabled
          ? "1px solid rgba(91, 228, 155, 0.3)"
          : mode === "dark"
          ? "1px solid #5BE49B"
          : "1.5px solid #5BE49B",
        background: props.disabled
          ? mode === "dark"
            ? "radial-gradient(72.6% 80.99% at 50% 50%, rgba(0, 0, 0, 0.30) 50.03%, rgba(91, 228, 155, 0.3) 100%), rgba(255, 255, 255, 0.05)"
            : "#e5e7eb"
          : mode === "dark"
          ? "radial-gradient(72.6% 80.99% at 50% 50%, rgba(0, 0, 0, 0.50) 50.03%, #5BE49B 100%), rgba(255, 255, 255, 0.10)"
          : "#5BE49B",
        color: props.disabled
          ? mode === "dark"
            ? "rgba(255, 255, 255, 0.5)"
            : "#b0b0b0"
          : "white",
        fontWeight: 500,
        fontSize: "15px",
        cursor: props.disabled ? "not-allowed" : "pointer",
        outline: "none",
        transition: "box-shadow 0.2s, background 0.2s, color 0.2s",
        boxShadow: props.disabled
          ? mode === "dark"
            ? "0 0 8px 0 rgba(91, 228, 155, 0.1)"
            : "0 0 8px 0 #e5e7eb"
          : mode === "dark"
          ? "0 0 16px 0 #5BE49B33"
          : "0 2px 8px 0 rgba(91,228,155,0.08)",
        opacity: props.disabled ? 0.6 : 1,
      }}
      className={cn(className)}
      {...buttonProps}
      onClick={handleClick}
    >
      <motion.div layout className="flex items-center gap-2">
        <Loader />
        <CheckIcon />
        <motion.span layout>{children}</motion.span>
      </motion.div>
    </motion.button>
  );
};

const Loader = () => {
  return (
    <motion.svg
      animate={{
        rotate: [0, 360],
      }}
      initial={{
        scale: 0,
        width: 0,
        display: "none",
      }}
      style={{
        scale: 0.5,
        display: "none",
        color: "white",
      }}
      transition={{
        duration: 0.3,
        repeat: Infinity,
        ease: "linear",
      }}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="loader"
    >
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M12 3a9 9 0 1 0 9 9" />
    </motion.svg>
  );
};

const CheckIcon = () => {
  return (
    <motion.svg
      initial={{
        scale: 0,
        width: 0,
        display: "none",
      }}
      style={{
        scale: 0.5,
        display: "none",
        color: "white",
      }}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="check"
    >
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M12 12m-9 0a9 9 0 1 0 18 0a9 9 0 1 0 -18 0" />
      <path d="M9 12l2 2l4 -4" />
    </motion.svg>
  );
};
