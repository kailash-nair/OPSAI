import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./index.css";

import Layout from "./Layout";
import UploadPage from "./pages/Upload";
import DashboardPage from "./pages/Dashboard";
import ArchivePage from "./pages/Archive";

const router = createBrowserRouter([
  { path: "/", element: <Layout><DashboardPage /></Layout> },
  { path: "/dashboard", element: <Layout><DashboardPage /></Layout> },
  { path: "/upload", element: <Layout><UploadPage /></Layout> },
  { path: "/archive", element: <Layout><ArchivePage /></Layout> },
]);

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
