(ns dev
  (:require [scicloj.clay.v2.api :as clay]))

(clay/make! {:format [:quarto :html]
             :base-source-path "notebooks"
             :source-path ["index.clj"
                           "utils.clj"
                           "llms.clj"]
             :base-target-path "docs"
             :book {:title "LLaMPPL.clj"}
             :clean-up-target-dir true})
