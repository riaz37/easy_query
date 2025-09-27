import * as React from "react";

const ReloadIcon = (props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="20"
    height="20"
    fill="none"
    viewBox="0 0 20 20"
    {...props}
  >
    <path
      fill={props.fill || "#fff"}
      d="M10 3.59V0L4.872 5.128 10 10.256V5.641c3.385 0 6.154 2.77 6.154 6.154s-2.77 6.154-6.154 6.154-6.154-2.77-6.154-6.154H1.795C1.795 16.308 5.487 20 10 20s8.205-3.692 8.205-8.205S14.513 3.59 10 3.59"
    ></path>
  </svg>
);

export default ReloadIcon;
