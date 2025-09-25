import * as React from "react";

const FilterIcon = (props: React.SVGProps<SVGSVGElement>) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="20"
    height="21"
    fill="none"
    viewBox="0 0 20 21"
    {...props}
  >
    <path
      fill={props.fill || "#fff"}
      d="M11.667 14.592a1.25 1.25 0 0 1 .12 2.494l-.12.006H8.334a1.25 1.25 0 0 1-.12-2.494l.12-.006zm2.5-5a1.25 1.25 0 1 1 0 2.5H5.834a1.25 1.25 0 0 1 0-2.5zm2.5-5a1.25 1.25 0 1 1 0 2.5H3.334a1.25 1.25 0 1 1 0-2.5z"
    ></path>
  </svg>
);

export default FilterIcon;
